from diffsynth.metrics import FIDMetric, ModelConfig
from modelscope import dataset_snapshot_download

metric = FIDMetric.from_pretrained(
    model_config=ModelConfig(model_id="DiffSynth-Studio/ImageMetrics", origin_file_pattern="FID/model.safetensors"),
    device="cuda",
)
score = metric.compute(
    "/mnt/nas3/sunyuzework/ImageGenRAG/data/inaturalist/gt/all",
    "/mnt/nas3/sunyuzework/ImageGenRAG/models/train/valfull/samples/test/step-186675_taxa",
)
print(f"FID score: {score:.3f}")