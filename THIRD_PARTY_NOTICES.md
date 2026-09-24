# Third-party notices

The sampling and latent preparation in `switching/flux.py` are adapted from
Hugging Face Diffusers' `pipeline_flux2_klein.py`:

https://github.com/huggingface/diffusers/blob/v0.37.0/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py

Copyright 2025 Black Forest Labs and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy at
https://www.apache.org/licenses/LICENSE-2.0 . A copy is included in
[`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt).

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.

Model checkpoints and dependencies retain their respective licenses and are not
bundled. New code is licensed under Apache-2.0; see [LICENSE](LICENSE).

The 4B notebook uses `examples/assets/imgedit_fox.png`, the
ImgEdit blue-bird-to-red-fox example saved with the paper. It is third-party
benchmark content, not covered by the new code's license. Reference: Yang Ye et
al., *ImgEdit: A Unified Image Editing Dataset and Benchmark* (2025),
https://arxiv.org/abs/2505.20275 . The publisher lists the dataset under Apache-2.0:
https://huggingface.co/datasets/sysuyy/ImgEdit . The accompanying text file contains the saved
edit instruction and target caption; the optional improved instruction in the
notebook is manually written for the demonstration.

The attention processors in `switching/selective.py` are adapted from Diffusers
`models/transformers/transformer_flux2.py` (v0.37.0), under the same copyright
and Apache-2.0 license above.

Context preparation and flow sampling in `switching/bagel.py` are adapted from
[ByteDance-Seed/Bagel](https://github.com/ByteDance-Seed/Bagel), revision
`a2fa77dd8caeefc41e6607ae0ec17408d3f4ee9f`.
Copyright 2025 Bytedance Ltd. and/or its affiliates. Licensed under Apache-2.0
(see the copy above). BAGEL checkpoints retain their publisher’s license.

The 9B and BAGEL notebooks use GEdit-Bench samples `750` and `535` in
`examples/assets/gedit_750.png` and `examples/assets/gedit_535.png`, respectively,
and their editing instructions. Target captions and improved editing instructions
accompany these demonstrations.
Source: [stepfun-ai/GEdit-Bench](https://huggingface.co/datasets/stepfun-ai/GEdit-Bench),
whose dataset card lists the MIT license. This third-party asset retains its
original ownership and license.

MIT License (GEdit-Bench asset):

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
