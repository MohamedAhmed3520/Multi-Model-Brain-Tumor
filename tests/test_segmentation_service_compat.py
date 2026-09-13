from app.models.segmentation_service import SegmentationService


def test_segmentation_service_can_load_checkpoint_with_missing_and_unexpected_keys():
    service = SegmentationService(model_path='./Tumor Models/my_checkpoint.pth')
    assert service.model is not None
    assert hasattr(service, 'model_name')
