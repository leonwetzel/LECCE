# TODO anders aanpakken omdat package textstat gebruikt wordt
# zie https://github.com/shivam5992/textstat voor meer info
class ReadabilityScore:
    def __init__(self):
        self.score = 0

    @classmethod
    def fleish(cls):
        return 0

    @classmethod
    def type_token_ratio(cls):
        return 0