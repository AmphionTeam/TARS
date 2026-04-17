import re
import json
from huggingface_hub import snapshot_download
from collections import defaultdict


def reward_len(completions, **kwargs):
    # 1024 for cot prompt
    TARGET_LEN = 1024
    # 512 for simple prompt
    # TARGET_LEN = 512
    return [
        -abs(TARGET_LEN - len(completion)) / TARGET_LEN for completion in completions
    ]


def xfinder_check(question, options, answer, llm_output):
    # Define input for a single evaluation
    standard_answer_range = []
    for a, o in zip("ABCD", options):
        standard_answer_range.append([a, str(o)])
    standard_answer_range = json.dumps(standard_answer_range)

    # Perform single example evaluation
    result = evaluator.evaluate_single_item(
        question, llm_output, standard_answer_range, "alphabet_option", answer
    )
    return result[-1]


evaluator = None


def reward_xfinder(completions, prompts, **kwargs):
    from xfinder.eval import Evaluator

    # Initialize the evaluator
    global evaluator
    if not isinstance(evaluator, Evaluator):
        print("initializing xfinder evaluator...")
        evaluator = Evaluator(
            model_name="xFinder-qwen1505",  # Model name
            inference_mode="local",  # Inference mode, 'local' or 'api'
            model_path_or_url=snapshot_download(
                "IAAR-Shanghai/xFinder-qwen1505"
            ),  # Anonymized model path or URL
            device="cuda",  # Device to run the model on
        )

    accuracy = []
    for completion, ques, opti, answ in zip(
        completions, kwargs["question"], kwargs["options"], kwargs["answer_index"]
    ):
        result = xfinder_check(ques, opti, "ABCD"[answ], completion)
        accuracy.append(result)
    return accuracy


embedding_model = None


def reward_align_audio_to_text(
    completions, prompts, __reward_xfinder__, question, **kwargs
):
    global embedding_model
    from sentence_transformers import SentenceTransformer, util

    # mix audio and text alignment reward.
    # if audio is None, then this is a text-only input, give 0 reward.
    # pair each text with audio, compute the similarity of the text and audio as audio's reward.

    # completions: list(str), text + audio
    assert (
        len(completions) % 2 == 0
    ), "Number of completions must be even. Each text completion should be paired with an audio completion."

    # assert "<|audio_1|>" in prompts[0], "First (even) prompt must contain audio."
    assert "<|audio_1|>" not in prompts[1], "Second (odd) prompt must be text-only."

    audio_completions = completions[0::2]
    audio_question = question[0::2]
    text_completions = completions[1::2]
    text_question = question[1::2]
    text_completions_correct = __reward_xfinder__[
        1::2
    ]  # only text completions' correctness
    assert len(audio_completions) == len(
        text_completions
    ), "Mismatched number of audio and text completions."

    if not embedding_model:
        print("initializing Qwen3-Embedding-0.6B...")
        embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

    audio_rewards = []
    rewards = []

    # Option 2. match audio with only correct text completions. and use max.
    # for audio_comp, audio_ques in zip(audio_completions, audio_question):
    #     max_align_score = 0  # initialize to lowest possible cosine similarity
    #     audio_emb = embedding_model.encode(audio_comp, convert_to_tensor=True)
    #     for text_comp, text_ques, text_correct in zip(text_completions, text_question, text_completions_correct):
    #         if audio_ques != text_ques: # only match with the same question
    #             continue
    #         if text_correct < 1:  # only match with correct text completions
    #             continue
    #         text_emb = embedding_model.encode(text_comp, convert_to_tensor=True)
    #         align_score = util.pytorch_cos_sim(audio_emb, text_emb).item() # -1 ~ 1
    #         if align_score > max_align_score:
    #             max_align_score = align_score
    #     audio_rewards.append(max_align_score)

    # Option 3. match audio with only correct text completions.
    available_text = defaultdict(list)
    for audio_comp, audio_ques in zip(audio_completions, audio_question):
        align_score = 0  # initialize with 0.
        audio_emb = embedding_model.encode(audio_comp, convert_to_tensor=True)

        # fisrt build available text_comp.
        for text_comp, text_ques, text_correct in zip(
            text_completions, text_question, text_completions_correct
        ):
            if audio_ques != text_ques:  # only match with the same question
                continue
            if text_correct < 1:  # only match with correct text completions
                continue
            available_text[audio_ques].append(text_comp)

        if len(available_text[audio_ques]) > 0:
            text_comp = available_text[audio_ques].pop(0)  # pop the first.
            text_emb = embedding_model.encode(text_comp, convert_to_tensor=True)
            align_score = util.pytorch_cos_sim(audio_emb, text_emb).item()  # -1 ~ 1
        audio_rewards.append(align_score)

    # Option 1. use avg_audio_rewards as text reward.
    avg_audio_rewards = sum(audio_rewards) / len(audio_rewards)
    for reward in audio_rewards:
        rewards.extend([reward, avg_audio_rewards])

    # Option 2. use 0 as text align reward.
    # for reward in audio_rewards:
    #     rewards.extend([reward, 0])

    # Option 3. use 1 as text align reward.
    # for reward in audio_rewards:
    #     rewards.extend([reward, 1])

    # Option 4. text correct => None, text wrong = 1.
    # for reward, is_correct in zip(audio_rewards, text_completions_correct):
    #     rewards.extend([reward, None if is_correct else 1])

    # Option 5. ingore all text.
    # for reward in audio_rewards:
    #     rewards.extend([reward, None])

    return rewards


import torch.nn.functional as F


class HiddenStateSimRewardORM:
    """
    Compute cosine similarity reward based on multi-layer hidden states.
    Assumes inputs are paired (Audio, Text) sequences, and computes reward only for pairs marked as correct.

    Args:
        target_layers (list): List of layer indices to compute similarity for (e.g., [1, 2, ..., 28]).
    """

    def __init__(self, target_layers=None):
        __name__ = "HiddenStateSimRewardORM"

        if target_layers is None:
            # FIXME: config target layers.
            self.target_layers = list(range(1, 33))  # all 32 layers i.e. (Soft, all)
            # self.target_layers = list(range(1, 11)) # first 10 layers (shallow)
            # self.target_layers = list(range(11, 21)) # middle 20 layers i.e. (Soft, middle)
            # self.target_layers = list(range(21, 31)) # deep 30 layers
            # self.target_layers = [31, 32] # last 2 layers
        else:
            self.target_layers = target_layers

        print(f"Processing Layers: {self.target_layers}")

    # Helper function: compute average cosine similarity across multiple layers
    def _calculate_avg_cos_sim(self, h_s_all, h_t_all) -> float:
        """Compute average cosine similarity of specified layers from two multi-layer hidden state tensors."""
        layer_cos_sims = []
        for layer_idx in self.target_layers:
            # Extract hidden states for specified layer ([N_L, N_seq + 1, H])
            h_s_l = h_s_all[layer_idx]  # [N_s + 1, H]
            h_t_l = h_t_all[layer_idx]  # [N_t + 1, H]

            c_h_s_l = h_s_l[
                1:-1, :
            ]  # Drop first token (prompt token) and last <|end|> token
            c_h_t_l = h_t_l[1:-1, :]  # Get hidden states corresponding to completion

            # Option 1: Sequence Aggregation (Mean Pooling)
            c_h_s_mean = c_h_s_l.mean(dim=0)  # [H]
            c_h_t_mean = c_h_t_l.mean(dim=0)  # [H]
            cos_sim = F.cosine_similarity(
                c_h_s_mean.unsqueeze(0), c_h_t_mean.unsqueeze(0)
            ).item()

            # Option 2: Last-Token - select last token to represent entire prompt completion sequence
            # Not as effective as entire sequence.
            # cos_sim = F.cosine_similarity(
            #     c_h_s_l[-1, :].unsqueeze(0),
            #     c_h_t_l[-1, :].unsqueeze(0)
            # ).item()

            layer_cos_sims.append(cos_sim)

        return sum(layer_cos_sims) / len(layer_cos_sims) if layer_cos_sims else 0.0

    def hidden_state_sim(self, completions, prompts, **kwargs):
        hidden_states_all = kwargs["hidden_states"]
        is_corrects = kwargs["__reward_xfinder__"]

        assert (
            len(hidden_states_all) == len(is_corrects) == len(completions)
        ), "Mismatched hidden states."
        # assert len(completions) == 8, "Currently only supports processing completions of same prompt at once, adjust batch_size and gradient_accumulation_steps"
        assert "<|audio_1|>" in prompts[0], "First (even) prompt must contain audio."
        assert "<|audio_1|>" not in prompts[1], "Second (odd) prompt must be text-only."

        rewards = [0.0] * len(completions)  # Initialize reward list

        # 1. Build pool of available correct text hidden states (implicitly grouped by pairs)
        # Assuming all correct matching Text Hidden States can serve as targets.
        correct_text_hiddens = []

        # Iterate all samples, separate correct Text Hidden States
        for i in range(1, len(completions), 2):  # Odd indices are Text
            idx_t = i
            if is_corrects[idx_t] >= 1:  # Only add correct text samples to pool
                h_t_all = hidden_states_all[idx_t]
                correct_text_hiddens.append(h_t_all)

        # 2. Compute rewards (only for Speech samples)
        num_correct_texts = len(correct_text_hiddens)
        match_index = 0

        for i in range(0, len(completions), 2):  # Even indices are Audio
            idx_s = i
            align_score = 0.0

            if num_correct_texts > 0:
                h_s_all = hidden_states_all[idx_s]

                # Select a Text Hidden State from pool
                h_t_match = correct_text_hiddens[match_index % num_correct_texts]

                # Compute multi-layer average cosine similarity
                align_score = self._calculate_avg_cos_sim(h_s_all, h_t_match)

                # Update matching index for next audio sample (looping)
                match_index += 1

            # Assign reward to Speech sample index
            rewards[idx_s] = align_score

        avg_reward = sum(rewards[0::2]) / len(rewards[0::2])
        for idx_t in range(1, len(completions), 2):
            rewards[idx_t] = avg_reward

        return rewards


think_pattern = re.compile(
    r"^<think>.*?<\/think>\s*<answer>.*The answer is [ABCD][\.:,].*<\/answer>$",
    re.DOTALL,
)
think_pattern_loose = re.compile(r"^<think>.*<answer>.*[ABCD][\.:,]", re.DOTALL)


def reward_format(completions, **kwargs):
    """
    Calculate reward score.

    Args:
        completions (list[str]): List of generated text.

    Returns:
        list[float]: List of reward scores for each text.
    """
    scores = []
    for text in completions:
        if think_pattern.match(text.strip()):
            scores.append(1.0)  # Matches format
        elif think_pattern_loose.match(text.strip()):
            scores.append(0.5)  # Roughly matches format
        else:
            scores.append(0.0)  # Does not match format
    return scores
