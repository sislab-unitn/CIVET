class GenerationCollator:
    def __init__(
        self,
        processor,
        instruction,
    ):
        self.processor = processor
        self.instruction = instruction

    def __call__(self, batch):

        samples = []
        questions = []
        images = []
        answers = []
        sample_ids = []

        for image, question, answer, sample_id in batch:
            samples.append(f"{self.instruction} {question}")
            questions.append(question)
            images.append(image)
            answers.append(answer)
            sample_ids.append(sample_id)

        # MOLMo for now cannot process multiple samples at once, just pop the first item since we will use it with batch_size = 1
        input_ids = self.processor.process(
            text=samples.pop(), images=images, return_tensors="pt", padding=True
        )

        questions = self.processor.tokenizer(
            questions, return_tensors="pt", padding=True
        )

        return input_ids, questions, answers, sample_ids
