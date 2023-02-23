class Registry:
    def __init__(self, name):
        self.name = name
        self.member = {}
    def register(self, src_type, dst_type):
        def func(f):
            self.member[(src_type, dst_type)] = f
            return f
        return func
    def get(self, src, dst):
        if not isinstance(src, type):
            src = type(src)
        if not isinstance(dst, type):
            dst = type(dst)
        return self.member[(src, dst)]
    def transform(self, src, dst):
        self.get(src, dst)(src, dst)
    def __repr__(self):
        return 'Registry: ' + self.name + " " + str(self.member)