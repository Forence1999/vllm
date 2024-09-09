# 08-14 naive beam+sample(not to select top1, use all sampled sequence)

from typing import Dict, List, Tuple, Union
import random
import math
import numpy as np
from vllm.config import SchedulerConfig
from vllm.core.scheduler import Scheduler
from vllm.engine.output_processor.interfaces import SequenceGroupOutputProcessor
from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.sequence import (
    Sequence,
    SequenceGroup,
    SequenceGroupOutput,
    SequenceOutput,
    SequenceStatus,
)
from vllm.transformers_utils.detokenizer import Detokenizer
from vllm.utils import Counter

logger = init_logger(__name__)


class SingleStepOutputProcessor(SequenceGroupOutputProcessor):
    """SequenceGroupOutputProcessor which handles "output processing" logic,
    which happens after the model returns generated token ids and before
    scheduling of the next batch. Output processing logic includes
    detokenization, and determining if a sequence is finished (e.g. via max len
    or eos token).

    The SingleStepOutputProcessor is specialized to the case where the model
    emits at most a single token per invocation, which precludes configurations
    such as speculative decoding or multi-step decoding. This enables beam
    search sampling, which requires forking/finishing/freeing sequences in a way
    that is currently difficult to schedule multiple steps ahead of time.
    """

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        detokenizer: Detokenizer,
        scheduler: Scheduler,
        seq_counter: Counter,
        stop_checker: StopChecker,
    ):
        self.scheduler_config = scheduler_config
        self.detokenizer = detokenizer
        self.scheduler = scheduler
        self.seq_counter = seq_counter
        self.stop_checker = stop_checker

    def process_outputs(
        self, sequence_group: SequenceGroup, outputs: List[SequenceGroupOutput]
    ) -> None:
        """Append all new tokens to sequences in the sequence group. Fork any
        surviving beam candidates; free any unsurviving ones.

        Invokes detokenizer to detokenize new tokens, and also marks sequences
        as finished if they meet stop conditions.
        """
        assert (
            len(outputs) == 1
        ), f"{type(self)} does not support multiple outputs per step"
        return self._process_sequence_group_outputs(sequence_group, outputs[0])

    def process_prompt_logprob(
        self, seq_group: SequenceGroup, outputs: List[SequenceGroupOutput]
    ) -> None:
        assert len(outputs) == 1, "Single step should only has 1 output."
        output = outputs[0]
        prompt_logprobs = output.prompt_logprobs
        if prompt_logprobs is not None:
            # auto-regression（持续把新的token添加到原来prompt后面）
            if seq_group.sampling_params.detokenize and self.detokenizer:
                # 原地改prompt_logprobs
                self.detokenizer.decode_prompt_logprobs_inplace(
                    seq_group, prompt_logprobs
                )
            if not seq_group.prompt_logprobs:
                # The first prompt token's logprob is None because it doesn't
                # have tokens that are precedent.
                seq_group.prompt_logprobs = [None]
            seq_group.prompt_logprobs.extend(prompt_logprobs)

    def _process_sequence_group_outputs(
        self, seq_group: SequenceGroup, outputs: SequenceGroupOutput
    ) -> None:
        # Process samples
        samples = outputs.samples
        parent_seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        existing_finished_seqs = seq_group.get_finished_seqs()
        parent_child_dict: Dict[int, List[SequenceOutput]] = {
            parent_seq.seq_id: [] for parent_seq in parent_seqs
        }
        for sample in samples:
            parent_child_dict[sample.parent_seq_id].append(sample)

        child_seqs: List[Tuple[Sequence, Sequence]] = []
        beam_width = seq_group.sampling_params.best_of
        #beam_width = 32
        #is_prompt = (
        #    len(existing_finished_seqs) == 0    # no finished seqs
        #    and len(parent_seqs) == 1           # only one parent seq, (the prompt)
        #    and len(samples) == initial_beam_width   # only beam_width seqs at the first step (implemented in /vllm/model_executor/layer/sampler.py)
        #)
        #if is_prompt:
        #    parent = parent_seqs[0]
        #    child_samples: List[SequenceOutput] = parent_child_dict[parent.seq_id]
        #    ancestor = 1
        #    for child_sample in child_samples[:-1]:
        #        new_child_seq_id: int = next(self.seq_counter)
        #        child = parent.fork(new_child_seq_id)
        #        child.data.ancestor = ancestor
        #        child.append_token_id(child_sample.output_token, child_sample.logprobs)
        #        child_seqs.append((child, parent))
        #        ancestor += 1
        #    last_child_sample = child_samples[-1]
        #    parent.data.ancestor = 0
        #    parent.append_token_id(last_child_sample.output_token, last_child_sample.logprobs)
        #    child_seqs.append((parent, parent))
        #else:
            # Process the child samples for each parent sequence
        for parent in parent_seqs:
            child_samples: List[SequenceOutput] = parent_child_dict[parent.seq_id]
            if len(child_samples) == 0:
                # This parent sequence has no children samples. Remove
                # the parent sequence from the sequence group since it will
                # not be used in the future iterations.
                parent.status = SequenceStatus.FINISHED_ABORTED
                seq_group.remove(parent.seq_id)
                self.scheduler.free_seq(parent)
                continue
            # Fork the parent sequence if there are multiple child samples.
            for child_sample in child_samples[:-1]:
                new_child_seq_id: int = next(self.seq_counter)
                child = parent.fork(new_child_seq_id)
                child.append_token_id(child_sample.output_token, child_sample.logprobs)
                child_seqs.append((child, parent))
            last_child_sample = child_samples[-1]
            parent.append_token_id(last_child_sample.output_token, last_child_sample.logprobs)
            child_seqs.append((parent, parent))

        for seq, _ in child_seqs:
            if seq_group.sampling_params.detokenize and self.detokenizer:
                new_char_count = self.detokenizer.decode_sequence_inplace(
                    seq, seq_group.sampling_params
                )
            else:
                new_char_count = 0
            self.stop_checker.maybe_stop_sequence(
                seq,
                new_char_count,
                seq_group.sampling_params,
                lora_req=seq_group.lora_request,
            )

        # Non-beam search case
        if not seq_group.sampling_params.use_beam_search:
            for seq, parent in child_seqs:
                if seq is not parent:
                    seq_group.add(seq)
                    if not seq.is_finished():
                        self.scheduler.fork_seq(parent, seq)
            for seq, parent in child_seqs:
                if seq is parent and seq.is_finished():
                    self.scheduler.free_seq(seq)
            return

        # Beam search case
        selected_child_seqs = []
        unselected_child_seqs = []

        length_penalty = seq_group.sampling_params.length_penalty

        existing_finished_seqs = [(seq, None, False) for seq in existing_finished_seqs]
        new_finished_seqs = [(seq, parent, True) for seq, parent in child_seqs if seq.is_finished()]
        all_finished_seqs = existing_finished_seqs + new_finished_seqs
        # len(child_seqs) + len(existing_finished_seqs) 一定小于等于 beam_width

        all_finished_seqs.sort(
            key=lambda x: x[0].get_beam_search_score(
                length_penalty=length_penalty, eos_token_id=x[0].eos_token_id
            ),
            reverse=True,
        )
        for seq, parent, is_new in all_finished_seqs[:beam_width]:
            if is_new:
                selected_child_seqs.append((seq, parent))
        for seq, parent, is_new in all_finished_seqs[beam_width:]:
            if is_new:
                unselected_child_seqs.append((seq, parent))
            else:
                seq_group.remove(seq.seq_id)

        running_child_seqs = [(seq, parent) for seq, parent in child_seqs if not seq.is_finished()]
        running_child_seqs.sort(
            key=lambda x: x[0].get_beam_search_score(
                length_penalty=length_penalty, eos_token_id=x[0].eos_token_id
            ),
            reverse=True,
        )

        # Check if we can stop the beam search.
        if len(running_child_seqs) == 0:
            stop_beam_search = True
        elif len(all_finished_seqs) < beam_width:
            stop_beam_search = False
        else:
            best_running_seq = running_child_seqs[0][0]
            current_worst_seq = all_finished_seqs[beam_width - 1][0]
            stop_beam_search = self._check_beam_search_early_stopping(
                seq_group.sampling_params.early_stopping,
                seq_group.sampling_params,
                best_running_seq,
                current_worst_seq,
            )

        if stop_beam_search:
            unselected_child_seqs.extend(running_child_seqs)
        else:
            selected_child_seqs.extend(running_child_seqs[:beam_width])
            unselected_child_seqs.extend(running_child_seqs[beam_width:])

        if len(unselected_child_seqs) != 0:
            print("len(unselected_child_seqs) != 0")

        for seq, parent in selected_child_seqs:
            if seq is not parent:
                seq_group.add(seq)
                if not seq.is_finished():
                    self.scheduler.fork_seq(parent, seq)
        for seq, parent in selected_child_seqs:
            if seq is parent and seq.is_finished():
                self.scheduler.free_seq(seq)
                #print("free selected seq that is finished")
                #if seq_group.num_children[seq.data.ancestor] == seq_group.num_finished_children[seq.data.ancestor]:
                #    print("All of this ancestor have finished, why another newly finished?")
                #seq_group.num_finished_children[seq.data.ancestor] += 1
                #print("seq_group.num_finished_children:",seq_group.num_finished_children)

        for seq, parent in unselected_child_seqs:
            if seq is parent:
                seq_group.remove(seq.seq_id)
                self.scheduler.free_seq(seq)
                #print("free unselected seq that is finished")
                #seq_group.num_finished_children[seq.data.ancestor] += 1
                #print("seq_group.num_finished_children:",seq_group.num_finished_children)

    def _process_sequence_group_outputs_forence(
        self, seq_group: SequenceGroup, outputs: SequenceGroupOutput
    ) -> None:
        assert (
            seq_group.sampling_params.use_beam_search
        ), "Must use beam for sampling_params.forence_params.num_candi_per_seq"
        print("seq_group :", len(seq_group.seqs_dict))
        # Process samples
        samples = outputs.samples
        parent_seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        existing_finished_seqs = seq_group.get_finished_seqs()
        #if len(existing_finished_seqs) > 0:
        #    print(len(existing_finished_seqs))
        
        parent_child_dict: Dict[int, List[SequenceOutput]] = {
            parent_seq.seq_id: [] for parent_seq in parent_seqs
        }
        for sample in samples:
            parent_child_dict[sample.parent_seq_id].append(sample)
        # List of (child, parent)
        child_seqs: List[Tuple[Sequence, Sequence]] = []

        # Process the child samples for each parent sequence
        for parent in parent_seqs:
            child_samples: List[SequenceOutput] = parent_child_dict[parent.seq_id]
            if len(child_samples) == 0:
                # This parent sequence has no children samples. Remove
                # the parent sequence from the sequence group since it will
                # not be used in the future iterations.
                parent.status = SequenceStatus.FINISHED_ABORTED
                seq_group.remove(parent.seq_id)
                self.scheduler.free_seq(parent)
                continue
            # Fork the parent sequence if there are multiple child samples.
            for child_sample in child_samples[:-1]:
                new_child_seq_id: int = next(self.seq_counter)
                child = parent.fork(new_child_seq_id)
                child.append_token_id(child_sample.output_token, child_sample.logprobs)
                child_seqs.append((child, parent))
                # FORENCE: add rank and logprob
                child.rank_forence = child_sample.logprobs[
                    child_sample.output_token
                ].rank
                child.logprob_forence = child_sample.logprobs[
                    child_sample.output_token
                ].logprob
            # Continue the parent sequence for the last child sample.
            # We reuse the parent sequence here to reduce redundant memory
            # copies, especially when using non-beam search sampling methods.
            last_child_sample = child_samples[-1]
            parent.append_token_id(
                last_child_sample.output_token, last_child_sample.logprobs
            )
            child_seqs.append((parent, parent))
            # FORENCE: add rank and logprob
            parent.rank_forence = last_child_sample.logprobs[
                last_child_sample.output_token
            ].rank
            parent.logprob_forence = last_child_sample.logprobs[
                last_child_sample.output_token
            ].logprob
        forence_params = getattr(seq_group.sampling_params, "forence_params", None)
        #rachel_params = getattr(seq_group.sampling_params, "rachel_params", None)
        length_penalty = seq_group.sampling_params.length_penalty
        
        # RACHEL: compute scores_forence for every new token (every new child_seq)
        #scores_forence = []
        #if (forence_params is not None) and (
        #    forence_params["mode"].lower() not in ["none", ""]
        #):
        #    sortfunc_forence = lambda x: x.get_beam_search_score_forence(
        #        length_penalty=length_penalty,
        #        eos_token_id=x.eos_token_id,
        #        forence_params=forence_params,
        #    )
        #else:
        #    sortfunc_forence = lambda x: x.get_beam_search_score(
        #        length_penalty=length_penalty, eos_token_id=x.eos_token_id
        #    )
        

        for seq, _ in child_seqs:
            if seq_group.sampling_params.detokenize and self.detokenizer:
                new_char_count = self.detokenizer.decode_sequence_inplace(
                    seq, seq_group.sampling_params
                )
            else:
                new_char_count = 0
            self.stop_checker.maybe_stop_sequence(
                seq,
                new_char_count,
                seq_group.sampling_params,
                lora_req=seq_group.lora_request,
            )
            #score_new_token = sortfunc_forence(seq) 
            #scores_forence.append(score_new_token)        
        #print(["{:.2f}".format(i) for i in scores_forence])
        # Beam search case
        # Select the child sequences to keep in the sequence group.
        selected_child_seqs = []
        unselected_child_seqs = []
        beam_width = seq_group.sampling_params.best_of
        length_penalty = seq_group.sampling_params.length_penalty
        all_indices = set(range(len(child_seqs)))

        is_prompt = (
            len(existing_finished_seqs) == 0  # no finished seqs
            and len(parent_seqs) == 1  # only one parent seq, (the prompt)
            and len(child_seqs) == beam_width  # only beam_width seqs at the first step (implemented in /vllm/model_executor/layer/sampler.py)
            and all(
                [len(seq.data.output_token_logprobs) == 1 for seq, _ in child_seqs]
            )  # all child seqs only have 1 token
        )
        if is_prompt:
            # only has top1 seqs
            top1_finished_indices = set(
                i for i, (child, _ ) in enumerate(child_seqs) if child.is_finished()
            )
            top1_running_indices = all_indices - top1_finished_indices
        else:
            # Process child sequences：
            # 1. select the top1 seqs
            # 2. divided into 4 catogories: top1_running_seqs, top1_finished_seqs, top1_eliminated_seqs, others
            # 3. the eliminated and others will be added to unselected_child_seqs

            # select the top beam_width sequences from the running
            # sequences for the next iteration to continue the beam
            # search.
            top1_indices = set()
            top1_parent_ids = set()
            #top1_eliminated_indices = set()
            # TODO: add critiaria to dertermine if top-1 should be eliminated 
            for i, (seq, parent) in enumerate(child_seqs):
                if (parent.seq_id not in top1_parent_ids):  # and seq.rank_forence == 1
                    #if scores_forence[i] < rachel_params['threshold'] and len(seq.data.output_token_ids) > rachel_params['ignore']:
                    #    pass
                    #else:
                    top1_indices.add(i)
                    top1_parent_ids.add(parent.seq_id)
            #if len(top1_indices) > rachel_params['num_keep']:
            #    for i, (seq, parent) in enumerate(child_seqs):
            #        if seq.rank_forence == 1 and (parent.seq_id not in top1_parent_ids) \
            #            and scores_forence[i] < rachel_params['threshold'] and len(seq.data.output_token_ids) > rachel_params['ignore']:
            #                print('\n', seq.seq_id,scores_forence[i], seq.output_text,'\n')
            #                top1_eliminated_indices.add(i)

            top1_running_indices = set(
                i for i in top1_indices if not child_seqs[i][0].is_finished()
            )
            top1_finished_indices = top1_indices - top1_running_indices
            #print("top1_indices: ", top1_indices)
            #print("top1_chi/par_ids: ", [(child_seqs[i][0].seq_id, child_seqs[i][1].seq_id) for i in top1_indices])
            #print("top1_eliminated_indices: ", top1_eliminated_indices)

            #if len(top1_running_indices) < beam_width:
                # FORENCE: not enough running sequences, try to sample more to augment
                #sampling_probs = [
                #    math.exp(seq.logprob_forence) for seq, _ in child_seqs      # exp(logprob) -> prob
                #]
                #sampling_indices = list(range(len(child_seqs)))
                # TODO: eliminate the parents whose top-1 is eliminated.
                #for i in top1_eliminated_indices:
                #for i in range(len(child_seqs)):
                #    tmp = np.exp(scores_forence[i]) 
                #    print("***", len(child_seqs[i][0].data.output_token_ids), '---', tmp, "***" )
                #    if len(child_seqs[i][0].data.output_token_ids) >= rachel_params['ignore']:
                #        if tmp <rachel_params['threshold']:
                #            print("<<< Throw: ", child_seqs[i][0].seq_id, '>>>')
                    #sampling_probs[i] = sampling_probs[i] / 100
                #print("sampling_probs: ", sampling_probs)
                #sampling_num = beam_width - len(top1_running_indices)
                #sampled_indices = random.choices(
                #    sampling_indices, weights=sampling_probs, k=sampling_num
                #)
                #sampled_running_indices = {
                #    i for i in sampled_indices if not child_seqs[i][0].is_finished()
                #}
                #top1_running_indices.update(sampled_running_indices - top1_indices)

        # Add all the sequences (except the finished and running ones) to the unselected list
        other_indices = all_indices - top1_running_indices - top1_finished_indices
        other_seqs = [child_seqs[i] for i in other_indices]
        if len(other_indices) != 0:
            print("other_indices =", len(other_indices))
        unselected_child_seqs.extend(other_seqs)

        # Process the finished sequences
        # Select the newly finished sequences with the highest scores to replace existing finished sequences.
        existing_finished_seqs = [(seq, None, False) for seq in existing_finished_seqs]

        new_finished_seqs = [
            (child_seqs[i][0], child_seqs[i][1], True) for i in top1_finished_indices
        ]
        all_finished_seqs = (
            existing_finished_seqs + new_finished_seqs
        )
        # Sort the finished sequences by their scores.
        all_finished_seqs.sort(
            key=lambda x: x[0].get_beam_search_score(
                length_penalty=length_penalty, eos_token_id=x[0].eos_token_id
            ),
            reverse=True,
        )
        for seq, parent, is_new in all_finished_seqs[:beam_width]:
            if is_new:
                # A newly generated child sequence finishes and has a high
                # score, so we will add it into the sequence group.
                selected_child_seqs.append((seq, parent))
        for seq, parent, is_new in all_finished_seqs[beam_width:]:
            if is_new:
                # A newly generated child sequence finishes but has a low
                # score, so we will not add it into the sequence group.
                # Additionally, if this sequence is a continuation of a
                # parent sequence, we will need remove the parent sequence
                # from the sequence group.
                unselected_child_seqs.append((seq, parent))
            else:
                # An existing finished sequence has a low score, so we will
                # remove it from the sequence group.
                seq_group.remove(seq.seq_id)

        # Process the running sequences
        running_child_seqs = [child_seqs[i] for i in top1_running_indices]
        # Sort the running sequences by their scores. # FORENCE
        running_child_seqs.sort(
            key=lambda x: x[0].get_beam_search_score(
                length_penalty=length_penalty, eos_token_id=x[0].eos_token_id
            ),
            reverse=True,
        )

        print("top1_running={}, existing_finish={}, new_finish={}, all_finish={}, sum={}".format(len(top1_running_indices), \
            len(existing_finished_seqs), len(new_finished_seqs), len(all_finished_seqs), len(top1_running_indices)+len(all_finished_seqs)))

        # Check if we can stop the beam search.
        if len(running_child_seqs) == 0:
            # No running sequences, stop the beam search.
            stop_beam_search = True
        elif len(all_finished_seqs) < beam_width:
            # Not enough finished sequences, continue the beam search.
            stop_beam_search = False
        else:
            # Check the early stopping criteria
            best_running_seq = running_child_seqs[0][
                0
            ]  # FORENCE FIXME: bug for local beam search
            current_worst_seq = all_finished_seqs[beam_width - 1][0]
            stop_beam_search = self._check_beam_search_early_stopping(
                seq_group.sampling_params.early_stopping,
                seq_group.sampling_params,
                best_running_seq,
                current_worst_seq,
            )

        if stop_beam_search:
            # Stop the beam search and remove all the running sequences from
            # the sequence group.
            unselected_child_seqs.extend(running_child_seqs)
        else:
            # Continue the beam search and select the top beam_width sequences
            # to continue the beam search.
            selected_child_seqs.extend(running_child_seqs[:beam_width])
            # The remaining running sequences will not be used in the next
            # iteration. Again, if these sequences are continuations of
            # parent sequences, we will need to remove the parent sequences
            # from the sequence group.
            unselected_child_seqs.extend(running_child_seqs[beam_width:])
        
        if len(unselected_child_seqs) != 0:
            print("unselected =", len(unselected_child_seqs))
        print("selected =", len(selected_child_seqs))

        # For newly created child sequences, add them to the sequence group
        # and fork them in block manager if they are not finished.
        for seq, parent in selected_child_seqs:
            if seq is not parent:
                seq_group.add(seq)
                if not seq.is_finished():
                    self.scheduler.fork_seq(parent, seq)

        # Free the finished and selected parent sequences' memory in block
        # manager. Keep them in the sequence group as candidate output.
        for seq, parent in selected_child_seqs:
            if seq is parent and seq.is_finished():
                self.scheduler.free_seq(seq)

        # Remove the unselected parent sequences from the sequence group and
        # free their memory in block manager.
        for seq, parent in unselected_child_seqs:
            if seq is parent:
                # Remove the parent sequence if it is not selected for next
                # iteration
                seq_group.remove(seq.seq_id)
                self.scheduler.free_seq(seq)
        print("")

    def _check_beam_search_early_stopping(
        self,
        early_stopping: Union[bool, str],
        sampling_params: SamplingParams,
        best_running_seq: Sequence,
        current_worst_seq: Sequence,
    ) -> bool:
        assert sampling_params.use_beam_search
        length_penalty = sampling_params.length_penalty
        if early_stopping is True:
            return True

        current_worst_score = current_worst_seq.get_beam_search_score(
            length_penalty=length_penalty, eos_token_id=current_worst_seq.eos_token_id
        )
        if early_stopping is False:
            highest_attainable_score = best_running_seq.get_beam_search_score(
                length_penalty=length_penalty,
                eos_token_id=best_running_seq.eos_token_id,
            )
        else:
            assert early_stopping == "never"
            if length_penalty > 0.0:
                # If length_penalty > 0.0, beam search will prefer longer
                # sequences. The highest attainable score calculation is
                # based on the longest possible sequence length in this case.
                max_possible_length = max(
                    best_running_seq.get_prompt_len() + sampling_params.max_tokens,
                    self.scheduler_config.max_model_len,
                )
                highest_attainable_score = best_running_seq.get_beam_search_score(
                    length_penalty=length_penalty,
                    eos_token_id=best_running_seq.eos_token_id,
                    seq_len=max_possible_length,
                )
            else:
                # Otherwise, beam search will prefer shorter sequences. The
                # highest attainable score calculation is based on the current
                # sequence length.
                highest_attainable_score = best_running_seq.get_beam_search_score(
                    length_penalty=length_penalty,
                    eos_token_id=best_running_seq.eos_token_id,
                )
        return current_worst_score >= highest_attainable_score
