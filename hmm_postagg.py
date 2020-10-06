'''
使用隐马尔科夫模型进行词性标注
'''
import numpy as np

#初始化字典
tag2id, id2tag = {},{}  #tag2id：词性的字典
word2id,id2word = {},{}

#建立字典
for line in open('./data/traindata.txt'):
    items = line.split('/')
    word = items[0]
    tag = items[1].rstrip()
    if word not in word2id:
        word2id[word] = len(word2id)
        id2word[len(word2id)] = word
    if tag not in tag2id:
        tag2id[tag] = len(tag2id)
        id2tag[len(tag2id)] = tag

# print(word2id)
# print(tag2id)


M = len(word2id)        #x有多少种情况（所有可能的观察数，显状态）
N = len(tag2id)         #y有多少种情况（有多少种隐状态）

P = np.zeros(N)     #初始状态矩阵
B = np.zeros((N,M)) #发射状态矩阵,N行M列
A = np.zeros((N,N)) #状态转移矩阵

#统计词频和tag的频数
prev_tag = ""       #前一个tag（使用中，若前一个tag为空，则认为当前tag为一个句子的开头位置）
for line in open('./data/traindata.txt'):
    items = line.split('/')
    wordId,tagId = word2id[items[0]],tag2id[items[1].rstrip()]
    if prev_tag == "":  #前一个tag为空，则当前词为句子开始的词
        P[tagId] += 1
        B[tagId][wordId] += 1
    else:       #中间的词，不用统计初始状态
        B[tagId][wordId] += 1
        # tag2id[prev_tag]：先将字符串prev_tag转换为对应的id。[tag2id[prev_tag]][tagId]：上一个tag id转移到当前tag id
        A[tag2id[prev_tag]][tagId] += 1

    if items[0] == '.':
        prev_tag = ''
    else:
        prev_tag = items[1].rstrip()

#归一化得到概率
P = P / sum(P)
for i in range(N):
    A[i] /= sum(A[i])
    B[i] /= sum(B[i])

def log(v):
    if v == 0:
        return np.log(v + 0.00001)
    return np.log(v)

#维特比算法解码，获得最优切分路径
def viterbi(x,P,A,B):
    x = [word2id[word] for word in x.split(' ')]    #x为输入的句子，转换为词对应的id，split(' ')：英文词之间是用空格隔开的
    T = len(x)
    dp = np.zeros((T,N))            # 动态规划组，保存每个词对应隐状态的概率值
    ptr = np.zeros((T,N),dtype=int) # 保存最大路径对应的下标

    for j in range(N):              # 边界条件考虑
        dp[0][j] = log(P[j]) + log(B[j][x[0]])  #第0个词属于第j种隐状态的概率，第0个词不考虑状态转移概率

    for i in range(1,T):        #遍历第i个词（第0个词在上面已经计算了）
        for j in range(N):      #计算当前词属于某个隐状态的概率
            dp[i][j] = -999999  #设置一个很小的值，用于后面比较大小，并保存最大值（后文有log运算，不要取0，-1之类的数，取一个很小的数）
            for k in range(N):  #求前一步属于某个隐状态时，当前词对应的隐状态概率最大
                # dp[i-1][k]：前一步为某个隐状态k的概率。A[k][j]：上一个隐状态k转移到当前隐状态j的概率
                # B[j][x[i]：当前隐状态j发射到当前显状态i的发射概率
                score = dp[i-1][k] + log(A[k][j]) + log(B[j][x[i]])     #当前词对应的概率
                if score > dp[i][j]:    # 取概率最大的
                    dp[i][j] = score    # 当前词i属于第j种隐状态的概率（如P24）
                    # 当前词i属于第j种状态时，其前一步中概率最大的对应的隐状态k（label）,(如P33对应tag3)
                    ptr[i][j] = k       #当前词i对应的前一个词i-1的隐状态（label）

    # 解码：求每个词对应的隐状态（取概率最大的）
    best_sequence = [0] * T     #保存每个词对应的概率最大的隐状态，即tag id
    best_sequence[T-1] = np.argmax(dp[T-1])     #最后一个词对应隐状态中概率最大的，dp[T-1]：最后一个词对应的所有隐状态
    for i in range(T-2,-1,-1):      #反向求得每一个词对应的隐状态
        # ptr记录的为前一步（前一个词）对应的隐状态（tag），ptr[i+1]才是第i个词
        # best_sequence[i+1]:根据第i+1个词对应的隐状态，去ptr中取第i个词对应的隐状态
        best_sequence[i] = ptr[i+1][best_sequence[i+1]]
    print(best_sequence)

    for i in range(len(best_sequence)):
        print(id2tag[best_sequence[i]])

if __name__ == "__main__":
    x = "I like to play ball"
    viterbi(x,P,A,B)

