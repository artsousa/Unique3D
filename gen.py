import os
import sys
import time
import argparse
import traceback
import numpy as np
from PIL import Image 
from pytorch3d.structures import Meshes
from app.utils import clean_up
from app.custom_models.mvimg_prediction import run_mvprediction #v2 also
from app.custom_models.normal_prediction import predict_normals
from scripts.multiview_inference import geo_reconstruct
from scripts.utils import save_glb_and_video
from scripts.refine_lr_to_sr import run_sr_fast


def generate3dv2(preview_img, input_processing, seed, out_dir="debug", render_video=True, do_refine=True, expansion_weight=0.1, init_type="std", debug=True):
    image_name = preview_img
    out_dir = os.path.join(out_dir, image_name.split(".")[0])
    if preview_img is None: raise Exception("preview_img is none")
    if isinstance(preview_img, str): preview_img = Image.open(preview_img)
    if preview_img.size[0] <= 512: preview_img = run_sr_fast([preview_img])[0]
    print(f"::image loaded: {image_name} shape: {preview_img.size} debug: {debug} out_dir: {out_dir}")
   
    try:
        st = time.time()
        rgb_pils, front_pil = run_mvprediction(preview_img, remove_bg=input_processing, seed=int(seed)) # 6s
        print("::run_mvprediction --- %s seconds ---" % (time.time() - st))
        if debug: 
            os.makedirs(out_dir, exist_ok=True)
            for i, _pil in enumerate(rgb_pils):
                _pil.save(os.path.join(out_dir, f"pil_{i}.png"), "PNG")
            front_pil.save(os.path.join(out_dir, f"front_pil.png"), "PNG")
            print(f"saved images: {len(rgb_pils)}: {rgb_pils[0].size} {front_pil.size}")
    except Exception as e:
        print(f"Failed to run mv_prediction: {e}")

    try:
        st = time.time()
        new_meshes = geo_reconstruct(
                rgb_pils, None, front_pil, do_refine=do_refine, predict_normal=True, expansion_weight=expansion_weight, init_type=init_type
        )
        print("::geo_reconstruct --- %s seconds ---" % (time.time() - st))
        if debug: print(f"new_meshes: {type(new_meshes)}")
    except Exception as e:
        ex_type, ex, tb = sys.exc_info()
        print(f"Failed to run geo_reconstruct {ex}")
        traceback.print_tb(tb)

    vertices = new_meshes.verts_packed()
    vertices = vertices / 2 * 1.35
    vertices[..., [0, 2]] = - vertices[..., [0, 2]]
    new_meshes = Meshes(verts=[vertices], faces=new_meshes.faces_list(), textures=new_meshes.textures)
    
    print(f"saving mesh...")
    ret_mesh, video = save_glb_and_video(
            out_dir, new_meshes, with_timestamp=True, dist=3.5, fov_in_degrees=2 / 1.35, cam_type="ortho", export_video=render_video
    )

    return ret_mesh, video


def main(args):
    print(vars(args))

    mesh, video = generate3dv2(
            preview_img=args.input_image, 
            input_processing=args.remove_bg, 
            seed=int(args.seed),
            render_video=args.render_video,
            do_refine=args.do_refine,
            expansion_weight=args.expansion_weight,
            init_type=args.init_type,
            debug=args.debug,
    )

    print(type(mesh), type(video))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog="3d Gen")

    parser.add_argument("--input-image", type=str, required=True)
    parser.add_argument("--remove-bg", action="store_true", default=True)
    parser.add_argument("--do-refine", action="store_true", default=True)
    parser.add_argument("--expansion-weight", type=float, default=0.1)
    parser.add_argument("--init-type", choices=["std", "thin"], default="std")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render-video", action="store_true")
    parser.add_argument("--debug", action="store_true", default=False)

    main(parser.parse_args())
