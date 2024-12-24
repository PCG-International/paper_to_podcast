import argparse
import os
from templates import enhance_prompt, initial_dialogue_prompt, plan_prompt
from dotenv import load_dotenv
import boto3
from langchain_core.output_parsers import StrOutputParser
from langchain_aws import ChatBedrock
from utils.script import generate_script, parse_script_plan
from utils.audio_gen import generate_podcast

# Load environment variables from a .env file
load_dotenv()

# Initialize the Bedrock client
bedrock_client = boto3.client(service_name='bedrock-runtime')

# Initialize the Bedrock model
llm = ChatBedrock(
    model_id=os.getenv("MODEL_ID"),
    client=bedrock_client,
    model_kwargs={
        "max_tokens": 2048,
        "temperature": 0.7,
    }
)

# chains
chains = {
    "plan_script_chain": plan_prompt | llm | parse_script_plan,
    "initial_dialogue_chain": initial_dialogue_prompt | llm | StrOutputParser(),
    "enhance_chain": enhance_prompt | llm | StrOutputParser(),
}


def main(pdf_path):
    # Step 1: Generate the podcast script from the PDF
    print("Generating podcast script...")
    script = generate_script(pdf_path, chains, llm)
    print("Podcast script generation complete!")

    print("Generating podcast audio files...")
    # Note: You'll need to modify generate_podcast to use Amazon Polly or another TTS service
    polly_client = boto3.client(service_name='polly')
    generate_podcast(script, polly_client)
    print("Podcast generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a podcast from a research paper."
    )
    parser.add_argument(
        "pdf_path", type=str, help="Path to the research paper PDF file."
    )

    args = parser.parse_args()
    main(args.pdf_path)
