from collections import defaultdict
def SwingRecall(u2items):
    u2Swing = defaultdict(lambda:dict())
    for u in u2items:
        wu = pow(len(u2items[u])+5,-0.35)
        for v in u2items:
            if v == u:
                continue
            wv = wu*pow(len(u2items[v])+5,-0.35)
            inter_items = set(u2items[u]).intersection(set(u2items[v]))
            for i in inter_items:
                for j in inter_items:
                    if j==i:
                        continue
                    if j not in u2Swing[i]:
                        u2Swing[i][j] = 0
                    u2Swing[i][j] += wv/(1+len(inter_items))
#         break
    return u2Swing