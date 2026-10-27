from pipecat.frames.frames import Frame, LLMContextFrame, MetricsFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.classifiers.base_classifier import ScoreQuestion, ClassifierError
from pipecat.classifiers.typesafe.jev.classifier import JevClassifier
from loguru import logger


class IntentClassifier(FrameProcessor):
    """Checks the mood of a user."""

    def __init__(self):
        super().__init__()
        self._classifier = JevClassifier(
            base_url="http://localhost:11434",
            api_key="ollama",
            model="laya",
        )
        self._classifier.add_event_handler("on_metrics", self._on_classifier_metrics)

    async def setup(self, setup: FrameProcessorSetup):
        await super().setup(setup)
        await self._classifier.setup(self.task_manager)

    async def cleanup(self):
        await super().cleanup()
        await self._classifier.cleanup()

    async def _on_classifier_metrics(self, classifier, data):
        await self.push_frame(MetricsFrame(data=data))

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            transcript = frame.context.messages  # extract messages as the state
            question = ScoreQuestion(
                instructions="How upset is the user?",
                levels=["calm", "frustrated", "angry"],
            )
            try:
                result = (await self._classifier.score(transcript, {"mood": question}))["mood"]
                angry_prob = result.probability("angry")
                frus_prb = result.probability("frustrated")
                logger.info(f"{transcript}\n    angry probability: {angry_prob} \n frustrated: {frus_prb}")
            except ClassifierError as e:
                logger.warning(f"Classifier error: {e}")

        await self.push_frame(frame, direction)