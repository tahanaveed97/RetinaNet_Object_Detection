import os
import shutil
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path')

    args = parser.parse_args()

    cwd = os.getcwd()

    dataset_folder_name = args.folder_path

    dataset_folder_path = os.path.join(cwd, dataset_folder_name)

    if os.path.exists(dataset_folder_path):
        for subdir, dirs, files in os.walk(dataset_folder_path):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    file_path = os.path.join(subdir, file)
                    shutil.copy(file_path, os.path.join(cwd, file))
        print("Images have been copied successfully.")
    else:
        print(f"The directory {dataset_folder_path} does not exist. Please ensure the dataset has been downloaded correctly.")


if __name__ == '__main__':
    main()
