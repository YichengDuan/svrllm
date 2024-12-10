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
    extract_interval: float,
    method: str = 'simple_mean',
    strength: int = 1,
    result_top_k: int = 1,
    video_path: str = "",
    tmp_frame_path: str = "",
    duration: float = 0.0,
    CC_json_path: str = "",
    VLM_BACKEND:VLM_EMB = VLM_BACKEND
) -> dict:
    """
    Evaluate video retrieval performance by calculating recall.

    :param extract_interval: Time interval (in seconds) between frame extractions.
    :param method: Retrieval method to use (e.g., 'simple_mean').
    :param strength: Strength parameter for retrieval (fixed at 1).
    :param result_top_k: Number of top results to consider for each retrieval (fixed at 1).
    :param video_path: Path to the input video file.
    :param tmp_frame_path: Directory to store extracted frames.
    :param duration: Total duration of the video (in seconds).
    :param CC_json_path: Path to the corresponding JSON file for captions or metadata.
    :param VLM_BACKEND: Endpoint or identifier for the Vision-Language Model backend.
    
    :return: Dictionary containing recall, correct count, and total frames.
    """
    try:
        # Cleanup databases
        cleanup_db()

        # Process the video to extract frames
        process_video(
            video_path=video_path,
            CC_json_path=CC_json_path,
            vlm_endpoint=VLM_BACKEND,
            video_extract_interval=extract_interval
        )

        # Extract frames from the video at specified intervals
        test_frames = extract_frames_opencv(
            video_path=video_path,
            output_dir=tmp_frame_path,
            interval=extract_interval,
            cv2backend=select_video_backend()
        )
        correct = 0
        total = len(test_frames)

        # Iterate through each frame to evaluate retrieval
        for frame in test_frames:
            image_path = frame["frame_path"]
            correct_time = frame["timestamp"]
            
            # Retrieve results using the specified method
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
            
            if frame_test_result:
                correct += 1

        # Calculate recall
        recall = correct / total if total > 0 else 0.0

        return {
            "recall": recall,
            "correct": correct,
            "total": total
        }

    except Exception as e:
        logging.error(f"Error in sv_test with parameters extract_interval={extract_interval}, method={method}: {e}")
        return {
            "recall": None,
            "correct": None,
            "total": None
        }

if __name__ == "__main__":
    # Configure logging
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sve_test_results")
    os.makedirs(results_dir, exist_ok=True)
    log_file_path = os.path.join(results_dir, "sv_experiment.log")
    logging.basicConfig(
        filename=log_file_path,
        filemode='w',
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Log the start of experiments
    logging.info("Starting video retrieval experiments.")

    # Define parameter lists
    extract_interval_list = [10, 25, 50, 100]  # in seconds
    method_list = ["simple_mean", "max_pool", "mean_pool", "concat_layers"]

    # Generate all possible combinations of parameters
    parameter_combinations = list(itertools.product(
        extract_interval_list,
        method_list
    ))

    # Initialize a list to store results
    results = []

    # Define paths and other constants
    local_path = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(local_path, "data/2023-01-01_1800_US_CNN_CNN_Newsroom_With_Fredricka_Whitfield.mp4")
    CC_json_path = os.path.join(local_path, "data/2023-01-01_1800_US_CNN_CNN_Newsroom_With_Fredricka_Whitfield.json")
    duration = calculate_video_duration(video_path)
    tmp_frame_path = os.path.join(local_path, "frames/svc_tmp/")

    # Iterate over each combination with a progress bar
    for extract_interval, method in tqdm(parameter_combinations, desc="Running Experiments"):
        cleanup_frames_directory(tmp_frame_path)
        logging.info(f"Running experiment with extract_interval={extract_interval}, method={method}")
        recall_result = sv_test(
            extract_interval=extract_interval,
            method=method,
            strength=1,        # Fixed value
            result_top_k=1,    # Fixed value
            video_path=video_path,
            tmp_frame_path=tmp_frame_path,
            duration=duration,
            CC_json_path=CC_json_path,
            VLM_BACKEND=VLM_BACKEND
        )

        # Append the result to the list
        results.append({
            "extract_interval": extract_interval,
            "method": method,
            "correct": recall_result["correct"],
            "total": recall_result["total"],
            "recall": recall_result["recall"]
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
    sns.set(style="whitegrid")

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

    # Plot 1: Recall vs. Extract Interval for each Method
    create_and_save_plot(
        df=df_results,
        x="extract_interval",
        y="recall",
        hue="method",
        title="Recall vs. Extract Interval for Different Methods",
        xlabel="Extract Interval (seconds)",
        ylabel="Recall",
        filename="Recall_vs_ExtractInterval.png"
    )

    # Plot 2: Recall vs. Method
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df_results,
        x="method",
        y="recall",
        ci=None
    )
    plt.title("Recall for Different Methods")
    plt.xlabel("Method")
    plt.ylabel("Recall")
    plt.tight_layout()

    # Save the plot
    plot2_path = os.path.join(results_dir, "Recall_vs_Method.png")
    plt.savefig(plot2_path)
    plt.close()
    logging.info(f"Plot saved to {plot2_path}")
    print(f"Plot saved to {plot2_path}")

    logging.info("All experiments completed successfully.")
    print("All experiments completed. Check the 'sve_test_results' directory for results and plots.")