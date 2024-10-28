#include <bits/stdc++.h>

#define ve vector
#define ld long double

using namespace std;

using pii = pair<int, int>;

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
int sgn(const ld& x) { return eq(x, 0) ? 0 : le(x, 0) ? -1 : 1; }

int sgn(vector<Point> &poly) {

}

struct edge {
    Point l, r;
    int up;
    edge() {}
    edge(Point a, Point b, int sw){
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
    if(abs(e1.x - e2.x) > eps)
        return e1.x < e2.x;
    if(abs(e1.type) != abs(e2.type))
        return abs(e1.type) < abs(e2.type);
    else return e1.type < e2.type;
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
    vector<int> bfs;
    for (int i = 0; i < n; ++i) {
        if (adj[i].size() == 1) {
            bfs.push_back(i);
        }
    }
    for (int pt = 0; pt < bfs.size(); ++pt) {
        int u = bfs[pt];
        int v = adj[u][0];
        adj[u].clear();
        for (auto &x : adj[v]) {
            if (x == u) {
                swap(adj[v].back(), x);
                adj[v].pop_back();
                break;
            }
        }
        if (adj[v].size() == 1) {
            bfs.push_back(v);
        }
    }
    for (int i = 0; i < n; i++) {
        used[i].assign(adj[i].size(), 0);
        auto compare = [&](int l, int r) {
            Point pl = vertices[l] - vertices[i];
            Point pr = vertices[r] - vertices[i];
            // cout << pl.cross(pr);
            if (pl.half() != pr.half())
                return pl.half() < pr.half();
            return pl.cross(pr) > eps;
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
            Point p1 = vertices[face[0]];
            ld sum = 0;
            for (int i = 0; i < face.size(); ++i) {
                Point p2 = vertices[face[i]];
                Point p3 = vertices[face[(i + 1) % face.size()]];
                sum += (p2 - p1).cross(p3 - p2);
            }
            if (sgn(sum) > 0)
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
            int prevsz = st.size();
            st.insert(e);
//            assert(prevsz == st.size() - 1);

        }
        else if(t == -1){
            st.erase(e);
        }
        else{
            auto it = st.upper_bound(e);
            if(it == st.begin()){
                //cout << e.l.x << " " << e.l.y << endl;
                continue;
            }
            it--;
            cnt[it->up]++;
        }
    }
}

vector<pii> init_edges(vector<vector<int>> &faces) {
    map<pii, vector<int>> mp;
    for (int i = 0; i < faces.size(); ++i) {
        auto &ve = faces[i];
        assert(ve.size() > 1);
        for (int j = 0; j < ve.size(); ++j) {
            int u = ve[j], v = ve[(j + 1) % ve.size()];
            mp[minmax(u, v)].push_back(i);
        }
    }
    vector<pii> edges;
    for (auto &[_, a] : mp) {
        sort(a.begin(), a.end());
        a.erase(unique(a.begin(), a.end()), a.end());
        assert(a.size() <= 2);
        if (a.size() > 1) {
            edges.emplace_back(a[0], a[1]);
        }
    }
    sort(edges.begin(), edges.end());
    edges.erase(unique(edges.begin(), edges.end()), edges.end());
    return edges;
}

mt19937 rng(5);

struct dsu {
    vector<int> p;
    dsu(int n): p(n, -1) {}
    int get(int u) {
        return p[u] < 0 ? u : p[u] = get(p[u]);
    }
    bool same(int u, int v) {
        return get(u) == get(v);
    }
    void unite(int u, int v) {
        u = get(u), v = get(v);
        if (p[u] < p[v]) {
            swap(u, v);
        }
        p[u] += p[v];
        p[v] = u;
    }
    void clear() {
        fill(p.begin(), p.end(), -1);
    }
};

const int L = 512, R = 1024;

int main(){
//    assert(freopen("small_Minsk.txt", "r", stdin));
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
    for(int i = 0; i < m; i++){
        int a, b;
        cin >> a >> b;
        a = realids[a]; b = realids[b];
        g[a].push_back(b);
        g[b].push_back(a);
    }
    ve<ve<int> > faces = find_faces(pts, g);
    for (auto &vec : faces) {
        for (auto &i : vec) {
            cout << "(" << pts[i].x << ", " << pts[i].y << ") ";
        }
        cout << "end\n";
    }
    int z = faces.size();
    int t;
    cin >> t;
    ve<Point> users(t);
    for(auto& i : users){
        int id;
        cin >> id;
        i.x = readlds();
        i.y = readlds();
    }
    return 0;
    ve<int> cnt(faces.size(), 0);
    scanline(pts, faces, cnt, users);
//    for(auto i : cnt)
//        cout << i << " ";
    auto edges = init_edges(faces);
    dsu d(z);
    vector<int> cur_cnt(z);
    int iter = 0;
    while (true) { // this part sucks
        shuffle(edges.begin(), edges.end(), rng);
        d.clear();
        cur_cnt = cnt;
        int cmp = z;
        for (auto &[u, v] : edges) {
            if (!d.same(u, v)) {
                int cu = cur_cnt[d.get(u)];
                int cv = cur_cnt[d.get(v)];
                if ((cu < L || cv < L) && cu + cv <= R) {
                    cur_cnt[d.get(u)] = cur_cnt[d.get(v)] = 0;
                    d.unite(u, v);
                    cur_cnt[d.get(u)] = cu + cv;
                }
            }
        }
        bool done = true;
        int mn = 10000000, mx = 0;
        for (int i = 0; i < z; ++i) {
            if (cur_cnt[d.get(i)] < L || cur_cnt[d.get(i)] > R) {
                done = false;
            }
            if (cur_cnt[d.get(i)] > 0) {
                mn = min(mn, cur_cnt[d.get(i)]);
            }
            mx = max(mx, cur_cnt[d.get(i)]);
        }
        if (done) break;
        cout << iter++ << ": " << mn << ", " << mx << "\n";
    }
    cout << "yay\n";
    return 0;
}
