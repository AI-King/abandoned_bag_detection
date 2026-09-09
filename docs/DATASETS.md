# Data sources and limitations

Source binaries stay outside Git. `configs/assets.lock.json` records source URLs
and SHA-256 hashes. `vision prepare` rejects changed or corrupted downloads.

| Asset | Purpose | Source |
| --- | --- | --- |
| COCO8 | Training smoke test | [Ultralytics COCO8](https://docs.ultralytics.com/datasets/detect/coco8/) |
| COCO128 | People and luggage development subset | [Ultralytics COCO128](https://docs.ultralytics.com/datasets/detect/coco128/) |
| LeftBag / LeftBag_PickedUp | Real video inference and qualitative review | [CAVIAR scenarios](https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/) |
| YOLO11n | Pretrained detector | [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) |

CAVIAR footage is publicly available under the page's Creative Commons BY-SA
terms. Acknowledge the EC Funded CAVIAR project/IST 2001 37540. These are INRIA lobby
clips, not railway-platform recordings. They last about a minute, much shorter
than the default five-minute threshold. The XML describes actors and activities;
we do not reinterpret a person's `leaving object` role as a bag bounding box.

COCO annotations have attribution terms; photographs retain individual licenses.
See the [COCO terms](https://cocodataset.org/#termsofuse) before redistributing images.
Downloaded assets are not relicensed by this repository. Ultralytics code and
weights have [AGPL/enterprise licensing](https://www.ultralytics.com/license).

`luggage-mini.yaml` remaps classes to person, backpack, handbag and suitcase.
Its deterministic split has 95 training and 33 validation images. Training has
5 backpacks, 15 handbags and 1 suitcase; validation has 1 backpack, 4 handbags and
3 suitcases. This exercises fine tuning but cannot establish station accuracy.
Image membership and counts are in `data/datasets/luggage-mini/split_manifest.json`.

Official COCO128 reuses its images for training and validation. Our derived split
uses disjoint image IDs, and the training audit rejects duplicate image content.
However, COCO-pretrained weights already saw these COCO training images. Scores
are development checks, not independent generalization estimates.

`vision fixture` creates `Synthetic_Timer_310s.mp4` from a cropped COCO suitcase
photo repeated at 1 fps for 310 seconds. A sidecar records source hash and crop.
It tests real YOLO inference through the 300-second timer; it contains no real
person-leaving event and must not be used as an accuracy benchmark.

For station training, label people, backpacks, handbags, suitcases, duffels,
sacks, shopping bags and relevant containers. Use `bag` for additional bag forms.
Include attended bags, occlusions, low light and hard negatives. Split by
camera/day/scene to avoid leakage from adjacent frames. Start with
`configs/custom-luggage.example.yaml`.
