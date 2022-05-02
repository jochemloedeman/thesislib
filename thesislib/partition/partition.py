import pickle


class Partition:
    def __init__(self, dynamic_vis_ids, static_vis_ids, dynamic_cap_ids, static_cap_ids,
                 dynamic_metadata, static_metadata):

        self.static = {"vis_ids": static_vis_ids,
                       "cap_ids": static_cap_ids,
                       "metadata": static_metadata}
        self.dynamic = {"vis_ids": dynamic_vis_ids,
                        "cap_ids": dynamic_cap_ids,
                        "metadata": dynamic_metadata}

        self.dynamic_fraction = len(dynamic_vis_ids) / (len(dynamic_vis_ids) + len(static_vis_ids))

    def save(self, file_path):
        with open(file_path, 'wb') as save_file:
            pickle.dump(self, save_file)

    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as partition_file:
            new_partition = pickle.load(partition_file)
        return new_partition