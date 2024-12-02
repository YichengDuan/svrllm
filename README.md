
# Subsequent Video Retrieval Enhancement using VLM and RAG

## Project Goal

The primary objective of this project is to retrieve subsequent TV shows and news videos based on an initial provided video clip. This involves enhancing the retrieval results through the integration of a Vision-Language Model (VLM) and a Retrieval-Augmented Generation (RAG) process.

---

## Overview

The project architecture is designed to process videos and their associated prompts using a Vision-Language Model (VLM), enhance video retrieval using a VectorDB, and subsequently store and manage the video data within a video file system. The approach focuses on ensuring that the retrieved results are enriched and contextually relevant.

---

## Approach

### System Architecture

1. **Input Sources:**
    - **Video:** User-provided video clip.
    - **Prompt:** User-provided textual input to guide the retrieval.

2. **Processing Components:**
    - **Video Pre-processing:** Initial processing of the video input to extract key frames, features, and metadata.
    - **Prompt Enhancing:** Processing the textual prompt to improve relevance and context for VLM-based queries.

3. **Core Model (VLM):**
    - The Vision-Language Model (VLM) processes both the video and enhanced prompt data, extracting meaningful features and generating representations.

4. **Retrieval and Storage:**
    - **VectorDB:** The processed data is stored in a Vector Database (VectorDB) for fast similarity search and retrieval.
    - **Video File System:** Subsequent retrieved video results are stored in a structured video file system for easy access and retrieval.

5. **Output Generation:**
    - **Answer/Summary:** Generates a summary or response based on the retrieved subsequent videos.

---

## Workflow

1. **User Input:**
    - The user provides a video and an optional prompt.
2. **Video Pre-processing:**
    - The video is processed to extract key data.
3. **Prompt Enhancing:**
    - The prompt is enhanced for better understanding by the VLM.
4. **Vision-Language Model Processing:**
    - The VLM processes the inputs and communicates with the VectorDB to identify related content.
5. **Retrieval and Management:**
    - Similar or related videos are retrieved and stored.
6. **Output Generation:**
    - A final summary or answer is provided based on the retrieved content.

---

## Technology Stack

- **Vision-Language Models (VLM)**
- **Vector Database (VectorDB)**
- **Retrieval-Augmented Generation (RAG) Process**
- **Video Processing Libraries**
- **Text Processing and NLP Tools**

---

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YichengDuan/svrllm.git
   ```
2. Install dependencies:
   ```bash
   cd SVRLLM
   pip install -r requirements.txt
   ```
3. Configure environment variables:
   - Configure any required API keys, database credentials in a `.config.yaml` file.

4. Run the project:
    processing video
   ```bash
   python test_main.py
   ```
    retrival testing
    ```bash
   python retrival.py
   ```
---

## Usage

1. Provide a video clip and optional prompt through the interface or API.
2. The system processes the input, enhances prompts, retrieves similar content, and returns results.
3. Access subsequent videos and summaries as output.

---

## Future Enhancements

- Integration with additional VLMs for broader context understanding.
- Improved RAG processes for enhanced response generation.
- Scalability improvements for large-scale video datasets.

---

## License

[MIT License](LICENSE)

---

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for more details.

---

## Contact

For any questions, issues, or feedback, please open an issue in this repository or contact the project maintainers.
