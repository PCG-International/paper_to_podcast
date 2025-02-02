# Paper-to-Podcast 🎤

**Paper-to-Podcast** is a tool that transforms academic research papers into an engaging and conversational podcast format. With this project, listeners can absorb the content of a research paper in a lively discussion involving three distinct personas—perfect for those who prefer listening over reading, especially during commutes or travel.

## Project Overview

### Objective

This app simulates a three-person discussion around the content of a research paper, making complex information more accessible and enjoyable to absorb. Instead of merely reading aloud, it converts papers into conversations that are engaging and intuitive, providing valuable insights and critical thinking.

### Personas

- **Host**: Guides the conversation, introducing each section and explaining the main points in an engaging and warm tone.
- **Learner**: Asks intuitive questions and brings curiosity to the discussion, helping listeners grasp core concepts.
- **Expert**: Provides in-depth knowledge and additional details, enhancing the discussion with profound insights.

This structure fosters an interactive listening experience, helping users better understand the paper in a way that feels natural and human.

### Code Structure and Key Components

- **Planning Chain**: Starts by creating a detailed plan for each section of the paper. Planning helps the model stay on track, reducing the chances of hallucinations or redundancy.
- **Discussion Chain**: Uses a retrieval-augmented generation model to expand on each section. This ensures the script stays true to the source content while generating meaningful dialogue.
- **Enhancement Chain**: Finalizes the script by removing redundancies, refining transitions, and ensuring a smooth flow.
- **Text-to-Speech**: The generated script is then converted into audio using the Amazon Polly API, producing realistic voices for each persona.

## Usage Instructions

### Prerequisites

1. Clone this repository:

   ```bash
   git clone https://github.com/PCG-International/paper_to_podcast.git
   ```

2. Move into the project directory:

   ```bash
   cd paper_to_podcast
   ```

3. Ensure you have run `aws configure` and setup the access to an AWS account with the proper IAM permissions.
4. Set your desired AWS options in the `.env` file.

*Example `.env` file:*

```bash
MODEL_ID="anthropic.claude-3-5-sonnet-20240620-v1:0"
EMBEDDINGS_MODEL_ID="amazon.titan-embed-text-v2:0"
AWS_REGION="us-east-1"
```

### Running the App

1. Place a research paper in PDF format in the project directory.
2. Run the script from the terminal, providing the path to your PDF file as an argument:

   ```bash
   python paper_to_podcast.py path/to/your/research_paper.pdf
   ```

### Sample Podcasts

You can find examples of podcasts generated using this pipeline in the `./sample_podcasts` directory.
