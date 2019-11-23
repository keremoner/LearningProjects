import numpy as np
class game:
    def __init__(self):
        self.table = np.zeros([9])
        self.turn = 1
        self.obsSpace = 9
        self.actSpace = 9
        self.time = 0
        self.totalTime = 0

    def reset(self):
        self.table = np.zeros([9])
        self.turn = 1
        self.time = 0
        self.totalTime = 0
        return self.table

    def render(self):
        print(self.table.reshape(3,3))

    def step(self, action):
        self.totalTime+=1
        if(self.table[action]!=0):
            reward = -1
            done = [False,False,False]
            return self.table, reward, done
        else:
            self.table[action] = self.turn
            self.time += 1
            done = self.isFinish()
            reward = 0
            self.turn *= -1
            if done[0] and done[0]:
                print("Bitti Yenen: " + str(-self.turn))
                print("Totaltime: " + str(self.totalTime))

            elif done[0] and (not done[1]):
                print("Bitti Berabere")
                print("Totaltime: " + str(self.totalTime))
            return self.table, reward, done




    def isFinish(self):
        newTable = self.table.reshape(3,3)
        rows = np.sum(newTable,axis=1)
        coloumns = np.sum(newTable, axis=0)

        for i in range(3):
            if rows[i] == 3 or coloumns[i] == 3 or rows[i] == -3 or coloumns[i] == -3:
                return [True,False,True]
        if (newTable[0][0] == 1 and newTable[1][1] == 1 and newTable[2][2] == 1) or (newTable[0][0] == -1 and newTable[1][1] == -1 and newTable[2][2] == -1) or (newTable[0][2] == 1 and newTable[1][1] == 1 and newTable[2][0] == 1) or (newTable[0][2] == -1 and newTable[1][1] == -1 and newTable[2][0] == -1):
            return [True,False,True]
        if self.time==9:
            return [True,True,True]
        return [False,False,True]

if __name__ == "__main__":
    game = env()
    state = game.reset()
    print(state)
    while True:

        action = int(input())
        newstate, reward, done = game.step(action)
        print(newstate)
        if done:
            state = game.reset()
        else:
            state = newstate