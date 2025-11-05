from qwen_vl_utils import process_vision_info


class GenerationCollator:
    def __init__(
        self,
        processor,
        instruction="",
    ):
        self.processor = processor
        self.instruction = instruction

    def __call__(self, batch):
        questions = []
        conversations = []
        texts = []
        answers = []
        sample_ids = []

        for image, question, answer, sample_id in batch:
            questions.append(question)
            answers.append(answer)
            sample_ids.append(sample_id)

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {
                            "type": "text",
                            "text": (
                                f"{self.instruction} {question}"
                                if self.instruction
                                else question
                            ),
                        },
                    ],
                }
            ]

            conversations.append(conversation)

        texts = [
            self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
            for msg in conversations
        ]
        image_inputs, video_inputs = process_vision_info(conversations)
        input_ids = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        questions = self.processor.tokenizer(  # type: ignore
            questions, return_tensors="pt", padding=True
        )

        return input_ids, questions, answers, sample_ids
