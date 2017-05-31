# Include & Network Lab - CNN Project
### 공공데이터 활용 CNN Project
>**공공데이터 및 Tensorflow를 활용한, CNN Project입니다.**

#### Project 세부내용
>**CNN 활용 주차장 관리 플랫폼 제작.**

#### License
MIT License

#### 저장소 참고사항
Pull requests는 매일 오전 10시 ~ 새벽 4시 사이에 확인합니다.
특이사항 공유는 저장소 내 Issues란에서 확인 부탁드립니다.

#### GitBash 사용 관련
git clone, init, status, add, pull, push, commit 등의 명령어는 기본적으로 개인 저장소에서 사용했으니 설명을 생략합니다.
>**git remote add upstream https://github.com/JNU-Include/CNN.git<br>
-upstream 저장소를 해당 url로 지정합니다.(최초 한번만 실행하도록 합시다.)<br>
-항상 저장소를 최신으로 유지하기 위함입니다.**<br>

**commit 전에 항상 해야 할 명령어**
>**git fetch upstream<br>
git merge upstream/master**

이후 status , add, commit 및 push하면 끝. (번거롭지만 타인의 프로젝트에 기여할 때엔 필수사항임에 따라, 연습겸 해보는 것으로 합시다.)<br>

**remote 설정과 관련해서 에러가 있을 경우, 즉, origin 주소가 개인 fork저장소가 아닌, JNU-Include-CNN 저장소로 지정되어 권한 없음으로 push가 안될 경우 해결법**

>**1)git bash 실행 후, cd CNN 입력<br>
2)git remote -v 입력 후, origin 주소 확인시,<br> https://github.com/JNU-Include/CNN.git 로 보인다면 수정 필요<br>
3)git remote set-url origin https:본인 fork 저장소 입력<br>
4)다시 git remote -v 입력 후 잘 변경되었는지 확인.**<br>
<br>

위 방법을 사용할 경우에 간단하게 해결될 문제임에 따라, 다시 clone하는 수고를 겪지 않았으면 좋겠습니다.<br>

**알아두면 나름 편리한(?) Git 명령어**<br>
>**1)explorer . - git bash에서 지정한 위치의 탐색기 창을 open**<br>
>**위 명령어는 윈도우 전용으로 확인되었습니다.**

git관련해서 이슈사항이 있으면 다시 정리해서 올려드리겠습니다.<br>

