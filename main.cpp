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

const ld eps = 1e-16;
bool le(const ld& a, const ld& b) { return a <= b; }
bool eq(const ld& a, const ld& b) { return abs(a - b) <= eps; }
int sgn(const ld& x) { return le(x, 0) ? eq(x, 0) ? 0 : -1 : 1; }


struct edge {
    Point l, r;
    int up;
    edge() {}
    edge(Point a, Point b, bool sw){
        l = a;
        r = b;
        up = sw;
    }
};

struct event{
    int type; //-1 — del, 0 — get, 1 - add
    ld x;
    edge e;
    int id;
};

inline bool operator <(const event& e1, const event& e2){
    if(!eq(e1.x, e2.x))
        return e1.x < e2.x;
    return e1.type > e2.type;
}


inline bool edge_cmp(const edge& e1, const edge&e2){
    Point a = e1.l, b = e1.r;
    Point c = e2.l, d = e2.r;
    int val = sgn(a.cross(b, c)) + sgn(a.cross(b, d));
    if (val != 0)
        return val > 0;
    val = sgn(c.cross(d, a)) + sgn(c.cross(d, b));
    return val < 0;
}

vector<vector<int>> find_faces(vector<Point> vertices, vector<vector<int>> adj) {
    int n = vertices.size();
    vector<vector<bool>> used(n);
    for (int i = 0; i < n; i++) {
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
            if (sign > 0)
                faces.emplace_back(face);
        }
    }
    return faces;
}

ld readlds(){
    string st;
    
    cin >> st;
    return stold(st);
}

int realids[MAXN];


void scanline(ve<Point> p, ve<ve<int> >& faces, ve<int>& cnt, ve<Point>& users){
    ve<event> es;
    for(int it = 0; it < faces.size(); it++){
        ve<int> &f = faces[it];
        int prev = f.back();
        for(auto i : f){
            if(p[prev].x < p[i].x){
                es.push_back({1, p[prev].x, edge(p[prev], p[i], it)});
                es.push_back({-1, p[i].x, edge(p[prev], p[i], it)});
            }
            prev = i;
        }
    }
    for(auto i : users)
        es.push_back({0, i.x, edge(i, i, 0)});
    sort(es.begin(), es.end());
    set<edge, decltype(*edge_cmp)> st(edge_cmp);
    for(int i = 0; i < es.size(); i++){
        auto [t, x, e, id] = es[i];
        if(t == 1){
            st.insert(e);
        }
        else if(t == -1){
            st.erase(e);
        }
        else{
            auto it = st.upper_bound(e);
            if(it == st.begin()){
                cout << e.l.x << " " << e.l.y << endl;
                continue;
            }
            it--;
            cnt[it->up]++;
        }
    }
}

int main(){
    ios_base::sync_with_stdio(0); cin.tie(0);
    
    int n;
    cin >> n;
    cout << n;
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
    for(int i = 0; i < m; i++){
        int a, b;
        cin >> a >> b;
        a = realids[a]; b = realids[b];
        g[a].push_back(b);
        g[b].push_back(a);
    }
    ve<ve<int> > faces = find_faces(pts, g);
    int t;
    cin >> t;
    ve<Point> users(t);
    for(auto& i : users){
        int id;
        cin >> id;
        i.x = readlds();
        i.y = readlds();
    }
    ve<int> cnt(faces.size(), 0);
    scanline(pts, faces, cnt, users);
}
