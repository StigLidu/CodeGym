def find_mode(score_lst):
    mode_lst = []
    max_nums = []
    
    for i in range(11):
        count = 0
        for score in score_lst:
            if (i == score):
                count += 1
                
        mode_lst.append(count)
    print(mode_lst)
    for  m in range(len(mode_lst)):
        if (max(mode_lst) == mode_lst[m]):
            max_nums.append(m)
            
    return max_nums
    

total_students = 20
scores = []

i = 1
while i <= total_students:
    score = int(input("Enter score: "))
    if (0 <= score <= 10):
        scores.append(score)
        i += 1
    else:
        print("Score is out of range.")

print("-"*5)
print("Original list:")   
print(scores)
print("Mode of scores:")
for num in find_mode(scores):
    print(num)