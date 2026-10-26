import os
import asyncio

from pipecat.classifiers.base_classifier import YesNoQuestion
from pipecat.classifiers.typesafe.jev.classifier import JevClassifier

async def main(): 
    classifier = JevClassifier(
        base_url="http://localhost:11434",
        api_key="ollama",
        model="laya"
    )

    results = await classifier.yes_no(
        "Hi, you've reached Dana. Leave a message.",
        {"voicemail": YesNoQuestion(instructions="is this a voicemail greeting?")},
    )

    print(results["voicemail"].is_yes)  # True
    print(results["voicemail"].probability)  # 0.97

if __name__ == "__main__":
    asyncio.run(main())