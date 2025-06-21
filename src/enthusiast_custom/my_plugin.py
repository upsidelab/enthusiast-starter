from enthusiast_common import ProductSourcePlugin

class MyPlugin(ProductSourcePlugin):
    def __init__(self, data_set_id, config: dict):
        super().__init__(data_set_id, config)

    def fetch(self) -> list:
        print("Fetch called")
        return []
