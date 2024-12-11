# Subsequent Video Retrieval Enhancement using VLM

## Table of Contents

- [Project Goal](#project-goal)
- [Overview](#overview)
- [Approach](#approach)
  - [System Architecture](#system-architecture)
- [Workflow](#workflow)
- [Technology Stack](#technology-stack)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Future Enhancements](#future-enhancements)
- [License](#license)
- [Contact](#contact)

---

## Project Goal

The primary objective of this project is to enhance the retrieval of subsequent TV shows and news videos based on an initially provided video clip. This is achieved by integrating a Vision-Language Model (VLM) with a Retrieval-Augmented Generation (RAG) process to improve the relevance and contextual accuracy of the retrieval results.

---

## Overview

This project leverages advanced machine learning techniques to process videos and associated textual prompts using a Vision-Language Model (VLM). By enhancing video retrieval with a Vector Database (VectorDB) and managing video data efficiently within a graph-based storage system, the approach ensures that retrieved results are both enriched and contextually relevant.

---

## Approach

### System Architecture

1. **Input Sources:**
    - **Video:** User-provided video clip.
    - **Prompt:** User-provided textual input to guide the retrieval process.

2. **Processing Components:**
    - **Video Pre-processing:** Extracts key frames, features, and metadata from the input video.
    - **Prompt Enhancement:** Processes the textual prompt to enhance relevance and context for VLM-based queries.

3. **Core Model (VLM):**
    - The Vision-Language Model (VLM) processes both video data and the enhanced prompt, extracting meaningful features and generating rich representations.

4. **Retrieval and Storage:**
    - **VectorDB:** Stores the processed data in a Vector Database for fast similarity searches and efficient retrieval.
    - **Graph DB:** Organizes and stores metadata in a structured Graph Database, facilitating easy access and retrieval of related video data.

5. **Output Generation:**
    - **Answer/Summary:** Generates a summary or response based on the retrieved subsequent videos, providing users with concise and relevant information.

---

## Workflow

1. **User Input:**
    - The user provides a video clip and an optional textual prompt.

2. **Video Pre-processing:**
    - The video is processed to extract key frames, features, and associated metadata.

3. **Prompt Enhancement:**
    - The textual prompt is enhanced to improve its relevance and context, ensuring better understanding by the VLM.

4. **Vision-Language Model Processing:**
    - The VLM processes the extracted video data and the enhanced prompt to generate vector embeddings.

5. **Retrieval and Management:**
    - Similar or related video frames are retrieved from the VectorDB and managed within the Graph Database.

6. **Output Generation:**
    - A final summary or answer is generated based on the retrieved content, providing users with meaningful insights.

---

## Technology Stack

- **Vision-Language Models (VLM)**
- **Vector Database (VectorDB)**
- **Retrieval-Augmented Generation (RAG) Process**
- **Video Processing Libraries**
---

## Setup and Installation

Follow the steps below to set up and run the project locally:

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/YichengDuan/svrllm.git
    ```

2. **Navigate to the Project Directory:**
    ```bash
    cd svrllm
    ```

3. **Install Dependencies:**
    Ensure you have Python installed (preferably Python 3.8 or higher). Install the required packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

4. **Configure Environment Variables:**
    - Rename the configuration template:
        ```bash
        cp config_template.yaml .config.yaml
        ```
    - Open `.config.yaml` and set the necessary API keys and database credentials as required.

5. **Prepare Data:**
    - Place your video files and corresponding transcript files into the `./data/` directory.
    - Clone the VLM model (e.g., Qwen2-VL-2B-Instruct) from Hugging Face into the `./model` directory:
        ```bash
        cd ./model
        pip install git+https://github.com/huggingface/transformers
        git clone https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
        ```

6. **Run Experiments:**
    - **Single Video Retrieval Experiment:**
        - Ensure you have the required video and transcript files (e.g., `2023-01-01_1800_US_CNN_CNN_Newsroom_With_Fredricka_Whitfield.json` and `2023-01-01_1800_US_CNN_CNN_Newsroom_With_Fredricka_Whitfield.mp4`) placed in the `./data/` directory.
        - Execute the experiments:
            ```bash
            # Run Single Video Retrieval Experiment in Strict Condition
            python single_video_experiment_strict.py

            # Run Single Video Retrieval Experiment in Loose Condition
            python single_video_experiment_loose.py
            ```


## Future Enhancements

- **Integration with Additional VLMs:**
    - Incorporate more Vision-Language Models to enhance context understanding and retrieval accuracy.

- **Improved RAG Processes:**
    - Refine the Retrieval-Augmented Generation pipeline to generate more accurate and contextually relevant responses.

- **Scalability Enhancements:**
    - Optimize the system to handle larger-scale video datasets, ensuring high performance and efficient retrieval times.

- **User Interface Development:**
    - Develop a user-friendly interface to facilitate easier interaction with the retrieval system.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For any questions, issues, or feedback, please:

- **Open an Issue:** Visit the [GitHub Issues](https://github.com/YichengDuan/svrllm/issues) page.

---