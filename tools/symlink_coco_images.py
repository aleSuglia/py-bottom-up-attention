import argparse
import os
import re
from glob import glob

parser = argparse.ArgumentParser(description="Python make_cococaption_id_names.py {cococaption basedir}\n\n"
                                             "After unpacking MS COCO zip files, you should end up with a\n"
                                             "base directory with train2017 and valid2017 folder in it.\n"
                                             "Provide this base dir as the first argument to this script and it\n"
                                             "will create symbolic links in the 'raw' folder, where each\n"
                                             "link name corresponds to the original image's ID.\n")

parser.add_argument("-image_dir", type=str, required=True, help="Input Image folder (WARNING absolute path)")
parser.add_argument("-image_subdir", nargs='+', default=["train2014", "val2014", "train2017", "val2017"],
                    help='Select the dataset subdir')
parser.add_argument("-data_out", type=str, required=True, help="Output symlink folder (WARNING absolute path)")

args = parser.parse_args()

assert args.image_dir.startswith(os.path.sep), "The path must be a root path: ".format(args.image_dir)
assert args.data_out.startswith(os.path.sep), "The path must be a root path: ".format(args.data_out)

for path in args.image_subdir:
    orig_path = os.path.join(args.image_dir, path)
    file_names = glob(os.path.join(orig_path, "*"))
    print(f"Processing {len(file_names)} files in folder {orig_path}")
    for name in file_names:
        filename = os.path.basename(name)
        # retrieve id images for COCO
        res = re.match(r'(\w*_\w*_)*0*(\d+.\w+)', filename)
        if not res:
            image_id, ext = os.path.splitext(filename)

            # ignore if it's not an image
            if ext.lower() not in (".jpg", ".png", ".jpeg"):
                continue

            image_filename = filename
        else:
            image_filename = res.group(2)

        try:
            # create symlink with id_image
            os.symlink(name, os.path.join(args.data_out, image_filename))
        except Exception as ex:
            print(f"Error for file {name}. Skipping it!")
