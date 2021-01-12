import numpy as np
import sys
import time
import multiprocessing
from functools import partial
import os

def DTW(A, share, Ms):     
    C = np.loadtxt("music_data\\" +Ms)  # 뮤직데이터 폴더 안에 뮤직데이터 파일 불러옴
    M, L = len(A), len(C)
    small = sys.maxsize     # small 최대값으로 초기화
    #for k in range(L-M):
    for k in range(1):  
        B=C[0+k:M+k]     #뮤직데이터 C를 허밍데이터만큼 자르고 B에 저장
        A, B = np.array(A), np.array(B) #A,B를 각각의 numpy 배열로 변환
        N = len(B)
        cost = sys.maxsize * np.ones((M, N))  #cost라는 배열에 (M,N) 크기의 배열에 1로 만들고 배열 전체를 최대값으로 초기화
                                                # cost 배열에는 각 좌표까지의 최단거리비용 값들이 저장될 것임
        # 첫번째 로우,컬럼 채우기
        cost[0, 0] = abs(A[0]- B[0])  #(0,0)에 리스트 A와 리스트 B의 첫 번째 값의 차의 절대값을 저장
        for i in range(1, M):   #각 행의 첫 번째 값 채우기
            cost[i, 0] =  abs(A[i]- B[0]) + cost[i - 1, 0]#B의 첫 번째 값과 A리스트에 해당하는 값들의 차의 절대값을 구하고 전 단계에서 구한 거리 값과 합친다.

        for j in range(1, N):   #첫 번째 행에 대해 채우기 
            cost[0, j] =abs(A[0]- B[j])+ cost[0, j - 1]
        
        # 나머지 행렬 채우기 
        for i in range(1, M):   #행 단위로 인덱스 1행부터 M행까지 연산 반복
            for j in range(1, N):   # i행의 1열부터 N열까지 반복 연산
                cost[i, j] = abs(A[i]- B[j])  + min(cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]) #현재 좌표의 차이값과 이전 거리 중 최솟값과 더함
        if cost[-1,-1] < small: # 최종 최단거리가 small 값보다 작으면 
            small = cost[-1,-1] # 최종 최단거리값을 small 저장
                                # 마지막 small에는 한 뮤직데이터의 최단거리 비용이 저장됨

    print("{0}파일의 리스트의 최단거리 : {1}".format(Ms,small))
    share[Ms] =small    #{"파일명" : "최단거리비용"} 이런 형식으로 저장
    print(len(share))   #share 크기 = 읽은 파일의 수
    
if __name__ == '__main__':  #인터프린터에서 직접 실행되고 있으면 __name__가  __main__인지 확인 후    True이면 다음 명령문 실행
    HD = os.listdir('humming_data') #허밍데이터 목록 불러옴
    print(HD)                       #허밍데이터 목록 출력
    hum = int(sys.stdin.readline()) #입출력으로 sys.stdin.readline() 사용하고 정수형으로 변환해서 hum에 저장 input보다 빠름
    A = np.loadtxt('humming_data\\'+HD[hum])    #허밍데이터폴더에 선택된 인덱스의 허밍데이터 파일 불러옴
    MD = os.listdir('music_data')               #뮤직데이터 목록 불러옴
    manager = multiprocessing.Manager()         #프로세서 객체 관리 변수 manager
    share = manager.dict()                      #서로 다른 프로세서들의 자원 공유 공간
    start_time = time.time()                    #현재 시간 저장
    p = multiprocessing.Pool(processes=4)       #사용할 프로세서 수
    func = partial(DTW, A)                      #함수를 사용할때 파라미터 값을 고정시켜줌
    p.starmap(func,[(share, Ms)for Ms in MD])   #map되는 함수가 여러 인자 값(튜플 인자)을 사용하기 위해 starmap을 사용
    p.close()                                   #프로세서 객체 닫음
    p.join()                                    #프로세서 객체들에 할당된 작업이 있으면 기다림
    five=sorted(share.items(), key = lambda item: item[1])  # 딕셔너리에서 value를 기준으로 오름차순 정렬
    print(five[:5])                             #오름차순으로 정렬되었기에 앞에는 value 값이 가장 작은거부터 5개 추출
    print("걸린 시간 : ",(time.time()-start_time)) #시작할 때 시간과 현재 시간의 차로 걸린 시간을 구함
