class DynamicMetadata:
    def __init__(self, dynamic_count, dynamic_verbs, static_verbs, lexnames):
        self.dynamic_count = dynamic_count
        self.dynamic_verbs = dynamic_verbs
        self.static_verbs = static_verbs
        self.lexnames = lexnames

    def get_all_tokens(self):
        return self.dynamic_verbs + self.static_verbs

