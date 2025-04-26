class RddFeatures:
    def __init__(self, **kwargs):
        valid_features = {'image'}
        for key, value in kwargs.items():
            if key in valid_features:
                setattr(self, key, value)
            else:
                raise ValueError(f'Invalid feature: {key}')
