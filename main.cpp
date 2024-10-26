#include <bits/stdc++.h>

#define ve vector
#define ld long double

using namespace std;

const int MAXN = 1e6+10;

struct Point {
    ld x, y;
    
    Point() {}
    Point(ld x_, ld y_): x(x_), y(y_) {}
    
    Point operator - (const Point & p) const {
        return Point(x - p.x, y - p.y);
    }
    
    ld cross (const Point & p) const {
        return x * p.y - y * p.x;
    }
    
    ld cross (const Point & p, const Point & q) const {
        return (p - *this).cross(q - *this);
    }
    
    int half () const {
        return int(y < 0 || (y == 0 && x < 0));
    }
};


vector<vector<int>> find_faces(vector<Point> vertices, vector<vector<int>> adj) {
    int n = MAXN;
    vector<vector<char>> used(n);
    for (int i = 0; i < n; i++) {
        used[i].resize(adj[i].size());
        used[i].assign(adj[i].size(), 0);
        auto compare = [&](int l, int r) {
            Point pl = vertices[l] - vertices[i];
            Point pr = vertices[r] - vertices[i];
            if (pl.half() != pr.half())
                return pl.half() < pr.half();
            return pl.cross(pr) > 0;
        };
        sort(adj[i].begin(), adj[i].end(), compare);
    }
    vector<vector<int>> faces;
    for (int i = 0; i < n; i++) {
        for (int edge_id = 0; edge_id < adj[i].size(); edge_id++) {
            if (used[i][edge_id]) {
                continue;
            }
            vector<int> face;
            int v = i;
            int e = edge_id;
            while (!used[v][e]) {
                used[v][e] = true;
                face.push_back(v);
                int u = adj[v][e];
                int e1 = lower_bound(adj[u].begin(), adj[u].end(), v, [&](int l, int r) {
                    Point pl = vertices[l] - vertices[u];
                    Point pr = vertices[r] - vertices[u];
                    if (pl.half() != pr.half())
                        return pl.half() < pr.half();
                    return pl.cross(pr) > 0;
                }) - adj[u].begin() + 1;
                if (e1 == adj[u].size()) {
                    e1 = 0;
                }
                v = u;
                e = e1;
            }
            reverse(face.begin(), face.end());
            int sign = 0;
            for (int j = 0; j < face.size(); j++) {
                int j1 = (j + 1) % face.size();
                int j2 = (j + 2) % face.size();
                ld val = vertices[face[j]].cross(vertices[face[j1]], vertices[face[j2]]);
                if (val > 0) {
                    sign = 1;
                    break;
                } else if (val < 0) {
                    sign = -1;
                    break;
                }
            }
            if (sign <= 0) {
                faces.insert(faces.begin(), face);
            } else {
                faces.emplace_back(face);
            }
        }
    }
    return faces;
}

ld readlds(){
    return 0;
//    string st;
    
//    cin >> st;
//    return stold(st);
}

int realids[MAXN];

int main(){
    ios_base::sync_with_stdio(0); cin.tie(0);
    int n;
    cin >> n;
    ve<Point> pts(n);
    for(int i = 0; i < n; i++){
        int id;
        cin >> id;
        realids[id] = i;
        pts[i].x = readlds();
        pts[i].y = readlds();
    }
    ve<ve<int> > g(n);
    int m;
    cin >> m;
    for(int i = 0; i < n; i++){
        int a, b;
        cin >> a >> b;
        a = realids[a]; b = realids[b];
        g[a].push_back(b);
        g[b].push_back(a);
    }
    ve<ve<int> > faces = find_faces(pts, g);
}
