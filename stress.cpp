#include <bits/stdc++.h>
#include <unistd.h>

using namespace std;

int main(){
    system("cd ../../build && make all");
    while(true){
        system("cd ../../build && ./generator 20 10 | ./main");
        cout <<"!" << endl;
    }
}
