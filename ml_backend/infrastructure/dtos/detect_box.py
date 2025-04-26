class DetectBox:
    """Helper class to store detection results during postprocessing."""

    def __init__(self, class_id, score, xmin, ymin, xmax, ymax):
        self.class_id = class_id
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
