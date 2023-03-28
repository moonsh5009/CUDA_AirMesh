# CUDA_Airmesh

논문 : https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11140493

기존 Airmesh의 문제점을 해결한 프로젝트

로프나 옷감 같은 부피가 없는 물체는 일정한 두께가 필요하다. 
이 두께가 두꺼울수록 삼각형이 유지해야하는 부피의 크기가 커지게 되면서 에지 플립이 되기 전에 불필요한 충돌이 발생하게 된다. 
이를 해결하기 위해 다음과 같은 새로운 삼각형 품질 계산 방법을 제안한다.

# Scene
#### 1
![scene1](https://user-images.githubusercontent.com/86860544/228161213-b7b527f6-eb5b-4d52-9c77-c218f9ff0b35.gif)

#### 2
![scene2](https://user-images.githubusercontent.com/86860544/228161177-bd466229-fea4-4edc-9a10-0207fccad028.gif)

#### 3
![scene3](https://user-images.githubusercontent.com/86860544/228161227-43046e4b-b988-4a5e-ac8a-bd68b3c3e2b6.gif)

# 참고문헌
 -  Müller, Matthias, Nuttapong Chentanez, Tae-Yong Kim, and Miles Macklin. "Air meshes for robust collision handling." ACM Transactions on Graphics (TOG) 34, no. 4 (2015): 1-9.
