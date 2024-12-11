import os
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
from main_pp import process_video
from retrieval import retrieve_method, VLM_BACKEND
from util import calculate_video_duration, extract_frames_opencv, perdict_result, cleanup_db
from vlm import VLM_EMB
import shutil
import cv2


def cleanup_frames_directory(directory_path: str) -> bool:
    """
    Deletes the specified directory and recreates it.

    :param directory_path: Path to the directory to be cleaned.
    :return: True if cleanup is successful, False otherwise.
    """
    try:
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path)
            logging.info(f"Successfully removed directory: {directory_path}")
        else:
            logging.warning(f"Directory does not exist: {directory_path}")

        # Recreate the directory
        os.makedirs(directory_path, exist_ok=True)
        logging.info(f"Recreated directory: {directory_path}")
        return True

    except Exception as e:
        logging.error(f"Error cleaning up directory {directory_path}: {e}")
        return False


def sv_test(
    strength: int,
    result_top_k: int,
    test_frames: list,
    duration: float = 0.0,
    method: str = "simple_mean"
) -> dict:
    """
    Evaluate video retrieval performance by calculating recall.

    :param strength: Strength parameter for retrieval.
    :param result_top_k: Number of top results to consider for each retrieval.
    :param video_path: Path to the input video file.
    :param tmp_frame_path: Directory to store extracted frames.
    :param duration: Total duration of the video (in seconds).
    :param CC_json_path: Path to the corresponding JSON file for captions or metadata.
    :param VLM_BACKEND: Endpoint or identifier for the Vision-Language Model backend.
    :param method: Retrieval method to use (e.g., 'simple_mean').

    :return: Dictionary containing recall, correct count, and total frames.
    """


    correct = 0
    total = len(test_frames[1::2])

    # Iterate through each frame to evaluate retrieval
    for frame in tqdm(test_frames[1::2], desc=f"Evaluating (strength={strength}, top_k={result_top_k})"):
        image_path = frame["frame_path"]
        correct_time = frame["timestamp"]
        # Retrieve results using the specified method and strength
        total_results = retrieve_method(
            image_path=image_path,
            strength=strength,
            total_time=duration,
            summary_output=False,
            method=method
        )

        # Determine if the retrieval result is correct
        frame_test_result = perdict_result(
            total_results=total_results,
            k=result_top_k,
            true_time=correct_time
        )

        if frame_test_result and correct_time % 15 != 0:  # Fixed extract_interval=15
            correct += 1

    # Calculate recall
    recall = correct / total if total > 0 else 0.0

    return {
        "recall": recall,
        "correct": correct,
        "total": total
    }


if __name__ == "__main__":
    # Configure logging
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sve_test_results")
    os.makedirs(results_dir, exist_ok=True)
    log_file_path = os.path.join(results_dir, "sv_experiment_loose.log")
    logging.basicConfig(
        filename=log_file_path,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Log the start of experiments
    logging.info("Starting video retrieval experiments.")

    # Fixed parameters
    extract_interval = 15  # Fixed to 15 seconds
    method = "simple_mean"  # Fixed to "simple_mean"
    result_top_k = 1       # Fixed to 1

    # Define experiment parameters
    strength_list = [1, 2, 3]

    # Initialize a list to store results
    results = []

    # Define paths and other constants
    local_path = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(local_path, "data/2023-01-01_1800_US_CNN_CNN_Newsroom_With_Fredricka_Whitfield.mp4")
    CC_json_path = os.path.join(local_path, "data/2023-01-01_1800_US_CNN_CNN_Newsroom_With_Fredricka_Whitfield.json")
    duration = calculate_video_duration(video_path)
    tmp_frame_path = os.path.join(local_path, "frames/svc_tmp/")

    # Cleanup frames directory
    cleanup_frames_directory(tmp_frame_path)
     # Cleanup databases
    cleanup_db()

    # Process the video once with fixed extract_interval and method
    logging.info(f"Processing video once with extract_interval={extract_interval}, method={method}")
    process_video(
        video_path=video_path,
        CC_json_path=CC_json_path,
        vlm_endpoint=VLM_BACKEND,
        video_extract_interval=extract_interval
    )

    # Extract frames once
    logging.info(f"Extracting frames with interval={extract_interval / 2.0} seconds")
    test_frames = extract_frames_opencv(
        video_path=video_path,
        output_dir=tmp_frame_path,
        interval=extract_interval / 2.0
    )

    # Iterate over each combination with a progress bar
    for strength in tqdm(strength_list, desc="Running Experiments"):
        logging.info(f"Running experiment with strength={strength}, result_top_k={result_top_k}")
        recall_result = sv_test(    
            test_frames=test_frames,
            strength=strength,
            result_top_k=result_top_k,
            duration=duration,
            method=method
        )

        # Append the result to the list
        results.append({
            "strength": strength,
            "result_top_k": result_top_k,
            "recall": recall_result["recall"],
            "correct": recall_result["correct"],
            "total": recall_result["total"]
        })

        # Log the result
        if recall_result['recall'] is not None:
            logging.info(f"Recall: {recall_result['recall']:.4f} (Correct: {recall_result['correct']}, Total: {recall_result['total']})")
        else:
            logging.warning("Recall could not be calculated due to an error.")
    # Convert results to a DataFrame
    df_results = pd.DataFrame(results)

    # Define the CSV file path
    csv_file_path = os.path.join(results_dir, "sv_test_results.csv")

    # Save the DataFrame to CSV
    df_results.to_csv(csv_file_path, index=False)
    logging.info(f"Results saved to {csv_file_path}")
    print(f"Results saved to {csv_file_path}")

    # Set Seaborn style for better aesthetics
    sns.set_theme(style="whitegrid")

    # Define a helper function to create and save plots
    def create_and_save_plot(df, x, y, hue, title, xlabel, ylabel, filename):
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=df,
            x=x,
            y=y,
            hue=hue,
            marker="o"
        )
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(title=hue)
        plt.tight_layout()
        plot_path = os.path.join(results_dir, filename)
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Plot saved to {plot_path}")
        print(f"Plot saved to {plot_path}")

    # Plot: Recall vs. Strength
    create_and_save_plot(
        df=df_results,
        x="strength",
        y="recall",
        hue=None,  # No hue since result_top_k is fixed
        title="Recall vs. Strength",
        xlabel="Strength",
        ylabel="Recall",
        filename="Recall_vs_Strength.png"
    )

    # Additionally, you can create a bar plot for better visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df_results,
        x="strength",
        y="recall",
        palette="viridis"
    )
    plt.title("Recall for Different Strength Values")
    plt.xlabel("Strength")
    plt.ylabel("Recall")
    plt.tight_layout()

    # Save the bar plot
    bar_plot_path = os.path.join(results_dir, "Recall_BarPlot_vs_Strength.png")
    plt.savefig(bar_plot_path)
    plt.close()
    logging.info(f"Bar plot saved to {bar_plot_path}")
    print(f"Bar plot saved to {bar_plot_path}")

    logging.info("All experiments completed successfully.")
    print("All experiments completed. Check the 'sve_test_results' directory for results and plots.")