

def pi():
    return 22/7


def crossProduct(vector1, vector2):
    if len(vector1) == 3 and len(vector2) == 3:
        i = vector1[1]*vector2[2] - vector2[1]*vector1[2]
        j = vector1[0]*vector2[2] - vector2[0]*vector1[2]
        k = vector1[0]*vector2[1] - vector2[0]*vector1[1]
        return [i, -j, k]
    if len(vector1) == 2 and len(vector2) == 2:

        k = vector1[0]*vector2[1] - vector2[0]*vector1[1]
        return [k]
    else:

        return "Argumentos no validos EN CROSS PRODUCT"


def subtract(vector1, vector2):
    newVector = []
    if len(vector1) != len(vector2):
        return "Argumentos no validos EN SUBSTRACT"
    else:
        for x in range(len(vector1)):
            newVector.append(vector1[x]-vector2[x])
        return newVector


def getVectorMagnitude(vector):
    totalSquared = 0
    for x in vector:
        totalSquared += pow(x, 2)
    return pow(totalSquared, .5)


def normalize(vector1):
    newVector = []
    magnitude = getVectorMagnitude(vector1)
    for x in vector1:
        if magnitude != 0:
            newVector.append(x/magnitude)
        else:
            newVector.append(int(x))
    return newVector


def dotProduct(vector1, vector2):
    if len(vector1) != len(vector2):
        return "Argumentos no validos en DOT PRODUCT"
    else:
        total = 0
        for x in range(len(vector1)):
            total += vector1[x]*vector2[x]
        return total


def deg2rad(data):

    pi = 22/7
    degree = data
    radian = degree*(pi/180)
    return radian


def multiplyMatrices(matrix1, matrix2):
    newMatrix = [[0 for i in range(len(matrix2[0]))]
                 for j in range(len(matrix1))]

    for col in range(len(newMatrix[0])):
        vec = [0 for j in range(len(matrix2))]
        for row in range(len(matrix2)):
            vec[row] = matrix2[row][col]
        for row in range(len(newMatrix)):
            d = dotProduct(matrix1[row], vec)
            newMatrix[row][col] = d
    return newMatrix


def add(vector1, vector2):
    newVector = []
    if len(vector1) != len(vector2):
        #print("ARGUMENTOS NO VALIDOS EN ADD")
        newVector = [0, 0, 0]
        return newVector
    else:
        for x in range(len(vector1)):
            newVector.append(vector1[x]+vector2[x])
        return newVector


def invertedAugmentMatrix(matrix):
    invertedAugmentMatrix = [[0.00 for i in range(
        len(matrix[0]))] for j in range(len(matrix))]
    for row in range(len(matrix)):
        for col in range(len(matrix[0])):
            invertedAugmentMatrix[row][col] = matrix[row][col]

    index = 0
    matrixWidth = len(invertedAugmentMatrix[0])
    for row in invertedAugmentMatrix:
        for i in range(matrixWidth):
            if i == index:
                row.append(1)
            else:
                row.append(0)
        index += 1
    return invertedAugmentMatrix


def scaleElements(matrix, scalar):
    if isinstance(matrix[0], list):
        for row in range(len(matrix)):
            for col in range(len(matrix[row])):
                matrix[row][col] = matrix[row][col]*scalar
    else:
        newVector = [0 for j in range(len(matrix))]
        for x in range(len(matrix)):
            if matrix[x] != 0:
                newVector[x] = matrix[x]*scalar

        return newVector


def inverse(matrix):
    if len(matrix) != len(matrix[0]):
        return 'Matrix no convertible'
    else:
        invertedMatrix = invertedAugmentMatrix(matrix)
        matrixWidth = len(matrix[0])
        row = 0
        curRow = 0
        col = 0
        lastPivRow = 0
        lastPivCol = 0

        while col < matrixWidth and curRow < len(invertedMatrix):
            if invertedMatrix[curRow][col] != 0:
                lastPivRow = curRow
                lastPivCol = col
                invertedMatrix[curRow] = scaleElements(
                    invertedMatrix[curRow], 1/(invertedMatrix[curRow][col]))
                row = curRow+1
                while row < len(invertedMatrix):
                    scalar = invertedMatrix[row][col] / \
                        invertedMatrix[curRow][col]
                    invertedMatrix[row] = add(scaleElements(
                        invertedMatrix[curRow], -scalar), invertedMatrix[row])
                    row += 1
                curRow += 1
            else:
                row = curRow+1
                while row < len(invertedMatrix):
                    if invertedMatrix[row][col] != 0:
                        temp = invertedMatrix[curRow]
                        invertedMatrix[curRow] = invertedMatrix[row]
                        invertedMatrix[row] = temp
                        col -= 1
                        break
                    row += 1
            col += 1

        col = lastPivCol
        curRow = lastPivRow

        while col != 0:
            if invertedMatrix[curRow][col] != 0:
                row = curRow-1
                while row != -1:
                    scalar = invertedMatrix[row][col] / \
                        invertedMatrix[curRow][col]
                    invertedMatrix[row] = add(scaleElements(
                        invertedMatrix[curRow], -scalar), invertedMatrix[row])
                    row -= 1
                curRow -= 1
            col -= 1

    for row in range(len(invertedMatrix)):
        invertedMatrix[row] = invertedMatrix[row][matrixWidth:]
    return invertedMatrix


def multiplyMatricesVector(matrix, vector):
    newMatrix = []
    for n in range(4):
        res = 0
        for j in range(4):
            res += matrix[n][j] * vector[j]
        newMatrix.append(res)
    return newMatrix


def ml(matrix, vector):
    newMatrix = []
    for n in range(3):
        res = 0

        res += matrix[n] * vector
        newMatrix.append(res)
    return newMatrix


def matriz_negada(matrix):
    newMatrix = []
    for i in range(len(matrix)):
        res = 0
        res += matrix[i] * -1
        newMatrix.append(res)
    return newMatrix
