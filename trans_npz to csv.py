import os
import numpy as np
import pandas as pd

def process_all(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    file_counter = 1

    for folder_name in sorted(os.listdir(input_dir)):
        folder_path = os.path.join(input_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        for fname in sorted(os.listdir(folder_path)):
            if not fname.lower().endswith('.npz'):
                continue

            file_path = os.path.join(folder_path, fname)
            with np.load(file_path) as data:
                # 1) grab components array
                if 'components' in data:
                    comp = data['components']
                elif 'component' in data:
                    comp = data['component']
                else:
                    print(f"⚠️  No 'components' key in {file_path}, skipping.")
                    continue

                # 2) grab timesteps array
                if 'timesteps' in data:
                    timesteps = data['timesteps']
                else:
                    print(f"⚠️  No 'timesteps' key in {file_path}, skipping.")
                    continue

            # shape checks
            if comp.ndim != 2 or comp.shape[1] != 5 or comp.shape[0] != timesteps.shape[0]:
                print(f"⚠️  Bad shapes in {file_path}: components {comp.shape}, timesteps {timesteps.shape}, skipping.")
                continue

            # 3) parse the two floats out of the filename
            base = os.path.splitext(fname)[0]  # e.g. "1.0130285502826442,0.7134383013818744"
            try:
                mob_str, grad_str = base.split(',')
                mobility = float(mob_str)
                gradient = float(grad_str)
            except Exception:
                print(f"⚠️  Can't parse two floats from '{base}', skipping.")
                continue

            # 4) build DataFrame
            df = pd.DataFrame(comp, columns=[f'comp_{i+1}' for i in range(5)])
            df.insert(0, 'step', timesteps)
            df['mobility'] = mobility
            df['gradient_coefficient'] = gradient

            # 5) write out as "1.csv", "2.csv", ...
            out_name = f"{file_counter}.csv"
            df.to_csv(os.path.join(output_dir, out_name), index=False)

            if file_counter % 500 == 0:
                print(f"→ Written {file_counter} files…")

            file_counter += 1

    print(f"✅ Done! {file_counter-1} files written to {output_dir}.")

if __name__ == "__main__":
    INPUT_DIR  = r"D:/sample2"
    OUTPUT_DIR = r"D:/sample3"
    process_all(INPUT_DIR, OUTPUT_DIR)
