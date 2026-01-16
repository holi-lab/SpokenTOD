"""Emotion tagger using OpenAI batch API."""

import concurrent.futures

from augmentation.constants import EMOTION_TOKENS, EMOTION_EXCLUDED
from augmentation.emotion.prompts import (
    build_emotion_prompt,
    build_batch_prompts,
    parse_emotion_response,
)
from augmentation.batch.client import BatchClient


class EmotionTagger:
    """Tag emotions for dialogue utterances using GPT."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4.1-mini",
        max_retries: int = 5,
        max_workers: int = 10,
    ):
        self.client = BatchClient(api_key=api_key, model=model, max_retries=max_retries)
        self.max_workers = max_workers

    def tag_utterances(
        self,
        utterances: list[str],
        wait: bool = True,
        use_batch: bool = True,
        max_workers: int | None = None,
    ) -> list[dict]:
        """Tag emotions for a batch of utterances.
        
        Args:
            utterances: List of user utterances to tag
            wait: Whether to wait for completion (ignored if use_batch=False)
            use_batch: Whether to use OpenAI Batch API or real-time calls
            max_workers: Number of workers for parallel real-time tagging (defaults to self.max_workers)
        
        Returns:
            List of {"label": int, "token": str} dicts
        """
        if not use_batch:
            workers = max_workers or self.max_workers
            
            def tag_single(utt):
                prompt = build_emotion_prompt(utt)
                response = self.client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=5
                )
                label = parse_emotion_response(response)
                token = EMOTION_TOKENS.get(label, "[neutral]")
                return {"label": label, "token": token}

            if workers <= 1:
                a = [tag_single(utt) for utt in utterances]
                print(f"Emotion tagging results: {a}")
                return a
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(tag_single, utterances))
            return results

        # Build batch prompts
        prompts = build_batch_prompts(utterances)
        
        # Submit batch
        batch_id = self.client.create_batch(prompts, "Emotion tagging batch")
        
        if not wait:
            return [{"batch_id": batch_id}]
        
        # Wait for completion
        self.client.wait_for_completion(batch_id)
        
        # Get and parse results
        results = self.client.get_results(batch_id)
        print(f"Batch {batch_id} raw results: {results}")
        parsed = self.client.parse_results(results)
        print(f"Batch {batch_id} results: {parsed}")
        
        # Map to emotion labels and tokens
        emotions = []
        for i in range(len(utterances)):
            custom_id = f"emotion-{i}"
            response = parsed.get(custom_id, "0")
            label = parse_emotion_response(response)
            token = EMOTION_TOKENS.get(label, "[neutral]")
            emotions.append({"label": label, "token": token})
        
        return emotions

    def should_tag(self, dataset: str) -> bool:
        """Check if dataset needs emotion tagging.
        
        Args:
            dataset: Dataset name
        
        Returns:
            True if tagging needed (not in EMOTION_EXCLUDED)
        """
        return dataset not in EMOTION_EXCLUDED


def tag_dialogue_emotions(
    turns: list[dict],
    dataset: str,
    tagger: EmotionTagger | None = None,
    existing_labels: list[int] | None = None,
) -> list[dict]:
    """Tag emotions for dialogue turns.
    
    Args:
        turns: List of turn dicts with "role" and "text"
        dataset: Dataset name
        tagger: EmotionTagger instance (created if None)
        existing_labels: Pre-existing labels (for EmoWOZ)
    
    Returns:
        List of turns with added emotion field
    """
    # If dataset already has labels, use them
    if existing_labels:
        result = []
        for i, turn in enumerate(turns):
            new_turn = dict(turn)
            if turn["role"] == "user" and i < len(existing_labels):
                label = existing_labels[i]
                if label >= 0:  # -1 means system turn
                    new_turn["emotion"] = {
                        "label": label,
                        "token": EMOTION_TOKENS.get(label, "[neutral]"),
                    }
            result.append(new_turn)
        return result
    
    # Skip if dataset excluded
    if dataset in EMOTION_EXCLUDED:
        return turns
    
    # Create tagger if needed
    if tagger is None:
        tagger = EmotionTagger(use_mock=True)
    
    # Collect user utterances
    user_indices = []
    user_texts = []
    for i, turn in enumerate(turns):
        if turn["role"] == "user":
            user_indices.append(i)
            user_texts.append(turn["text"])
    
    # Tag emotions
    emotions = tagger.tag_utterances(user_texts)
    print(f"Tagged emotions: {emotions}")
    # Apply to turns
    result = list(turns)
    for idx, emotion in zip(user_indices, emotions):
        result[idx] = dict(result[idx])
        result[idx]["emotion"] = emotion
    
    return result
