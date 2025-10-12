class PaperRouter:
    def __init__(self): self.orders = []
    def send(self, child_order):
        self.orders.append(child_order)
        return {'status':'accepted','id':len(self.orders)}
