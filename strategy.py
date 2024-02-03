class Strategy:
    def S1(self, change):
        if change>0:
            return "long"
        else:
            return "short"
        
    def S2(self, change_1, change_2):
        if change_1<0:
            if change_2<0:
                if change_1>change_2:
                    return f"short 1"
                else:
                    return f"short 2"
            else:
                if abs(change_1)>abs(change_2):
                   return f"short 1"
                else:
                    return f"long 2"
        else:
            if change_2<0:
                if abs(change_1)>abs(change_2):
                    return f"long 1"
                else:
                    return f"short 2"
            else:
                if change_1>change_2:
                    return f"long 1"
                else:
                    return f"long 2"
                
    def S3(self, change_1, change_2):
        if change_1<0:
            if change_2<0:
                return f"short 1, short 2"
            else:
                return f"short 1, long 2"
        else:
            if change_2<0:
                return f"long 1, short 2"
            else:
                return f"long 1, long 2"



                

        
