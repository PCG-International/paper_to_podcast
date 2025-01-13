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
bedrock_client = boto3.client(
    service_name='bedrock-runtime', region_name=os.getenv("AWS_REGION"))

# Initialize the Bedrock model
llm = ChatBedrock(
    model_id=os.getenv("MODEL_ID"),
    client=bedrock_client,
    model_kwargs={
        "max_tokens": 4096,
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
    # If a script has already been generated, skip the generation process
    if os.path.exists(f"./script_{os.path.basename(pdf_path).replace('.pdf', '.txt')}"):
        print("Podcast script already exists. Skipping generation process...")
        script_path = f"./script_{os.path.basename(
            pdf_path).replace('.pdf', '.txt')}"
        with open(script_path, "r", encoding="utf-8") as file:
            script = file.read()
    else:
        # Step 1: Generate the podcast script from the PDF
        print("Generating podcast script...")
        script = generate_script(pdf_path, chains, llm)
        print("Podcast script generation complete!")

        # write the script to a file
        script_path = f"./script_{os.path.basename(
            pdf_path).replace('.pdf', '.txt')}"
        with open(script_path, "w", encoding="utf-8") as file:
            file.write(script)

    print("Generating podcast audio files...")
    polly_client = boto3.client(
        service_name='polly', region_name=os.getenv("AWS_REGION"))
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
