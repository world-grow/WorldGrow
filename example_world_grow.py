import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import imageio
from trellis.pipelines import WorldGrowPipeline
from trellis.utils import render_utils

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = WorldGrowPipeline.from_pretrained("UranusITS/WorldGrow")
pipeline.cuda()

# Run the pipeline
world_size = (3, 3) # Specify the desired world size (in blocks)
outputs = pipeline.run(world_size=world_size)

# Render the outputs
r = 1.5 + max(world_size)
look_at = [0.25 * (world_size[0] - 1), 0.25 * (world_size[1] - 1), 0]
video = render_utils.render_video(outputs['gaussian'], r=r, look_at=look_at)['color']
imageio.mimsave("sample.mp4", video, fps=30)

outputs['mesh'].export("sample.glb")
outputs['gaussian'].save_ply("sample.ply")
