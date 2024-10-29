#include <bits/stdc++.h>
#define ve vector
#define ll long long

using namespace std;

using pii = pair<int, int>;

const int MAXN = 1e6+10;

struct Point {
    ll x, y;

    Point() {}
    Point(ll x_, ll y_): x(x_), y(y_) {}

    Point operator - (const Point & p) const {
        return Point(x - p.x, y - p.y);
    }

    __int128 cross (const Point & p) const {
        return (__int128)x * p.y - (__int128)y * p.x;
    }

    __int128 cross (const Point & p, const Point & q) const {
        return (p - *this).cross(q - *this);
    }

    int half () const {
        return int(y < 0 || (y == 0 && x < 0));
    }
};

#define le(a, b) (a <= b)
#define eq(a, b) (a == b)

int sgn(const __int128& x) { return eq(x, 0) ? 0 : le(x, 0) ? -1 : 1; }


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
    ll x;
    edge e;
};

inline bool operator <(const event& e1, const event& e2){
    if(!eq(e1.x, e2.x))
        return e1.x < e2.x;
    if(abs(e1.type) != abs(e2.type))
        return abs(e1.type) < abs(e2.type);
    else if(e1.type != e2.type)
        return e1.type < e2.type;
    else return e1.e.up < e2.e.up;
}


inline bool edge_cmp(const edge& e1, const edge&e2){
    Point a = e1.l, b = e1.r;
    Point c = e2.l, d = e2.r;
    int val = sgn(a.cross(b, c)) + sgn(a.cross(b, d));
    if (val != 0)
        return val > 0;
    val = sgn(c.cross(d, a)) + sgn(c.cross(d, b));
    if(val != 0)
        return val < 0;
    return e1.up < e2.up;
}

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
        if (u == v) return;
        if(p[u] > p[v]) swap(u, v);
        p[u] += p[v];
        p[v] = u;
        
    }
    void clear() {
        fill(p.begin(), p.end(), -1);
    }
};

void delete_inner(const ve<Point>& p, ve<ve<int> >& faces){
    int n = p.size();
    ve<event> es;
    dsu ds(n);
    for(auto f : faces){
        ds.unite(f.front(), f.back());
        for(int i = 1; i < f.size(); i++)
            ds.unite(f[i], f[i - 1]);
    }
    
    for(int it = 0; it < faces.size(); it++){
        ve<int> &f = faces[it];
        int prev = f.back();
        
        for(auto i : f){
            if(p[prev].x < p[i].x){
                es.push_back({1, p[prev].x, edge(p[prev], p[i], ds.get(i) + 1)});
                es.push_back({-1, p[i].x, edge(p[prev], p[i], ds.get(i) + 1)});
            }
            else if(p[prev].x > p[i].x){
                es.push_back({-1, p[prev].x, edge(p[i], p[prev], -ds.get(i) - 1)});
                es.push_back({1, p[i].x, edge( p[i], p[prev], -ds.get(i) - 1)});
            }
            prev = i;
        }
    }
    sort(es.begin(), es.end());
    ve<bool> del(n);
    set<edge, decltype(*edge_cmp)> st(edge_cmp);
    for(auto [tp, x, e] : es){
        if(del[abs(e.up) - 1]) continue;
        if(tp == 1){
            auto it = st.insert(e).first;
            if(it == st.begin()) continue;
            auto prv = prev(it);
            while(prv != st.begin() && del[abs(prv->up) - 1]){
                prv = prev(st.erase(prv));
            }
            if(it->up > 0 && prv->up > 0){
                del[abs(it->up) - 1] = 1;
            }
        }
        else if(tp == -1)
            st.erase(e);
    }
    ve<ve<int> > nfaces;
    for(int i = 0; i < faces.size(); i++)
        if(!del[ds.get(faces[i][0])])
            nfaces.push_back(faces[i]);
    faces.swap(nfaces);

}

vector<vector<int>> find_faces(vector<Point> vertices, vector<vector<int>> adj, bool inner) {
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
                assert(e1 != adj[u].size() + 1);
                v = u;
                e = e1;
            }
            reverse(face.begin(), face.end());
            int sign = 0;
            Point p1 = vertices[face[0]];
            __int128 sum = 0;
            for (int i = 0; i < face.size(); ++i) {
                Point p2 = vertices[face[i]];
                Point p3 = vertices[face[(i + 1) % face.size()]];
                sum += (p2 - p1).cross(p3 - p2);
            }
            if ((!inner && sgn(sum) > 0) || (inner && sgn(sum) < 0))
                faces.emplace_back(face);
        }
    }
    return faces;
}

ll readlds(){
    string st;
    cin >> st;
    size_t pos = st.find(".");
    if(pos == -1)
        pos = st.size();
    else st.erase(pos, 1);
    while(st.size() - pos < 15)
        st.push_back('0');
    
    return stoll(st);
}

int realids[MAXN];
int inputids[MAXN];


void scanline(const ve<Point>& p, ve<ve<int> >& faces, ve<int>& cnt, ve<Point>& users){
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
        auto [t, x, e] = es[i];
        if(t == 1){
            st.insert(e);
        }
        else if(t == -1){
            st.erase(e);
        }
        else{
            auto it = st.upper_bound(e);
            if(it == st.begin()){cout << "!"; continue;
            }
//            assert(it != st.begin());
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


const int L = 512, R = 1024;


int main(){
//    assert(freopen("small_Minsk.txt", "r", stdin));
    int n;
    cin >> n;
    ve<Point> pts(n);
    for(int i = 0; i < n; i++){
        int id;
        cin >> id;
        inputids[i] = id;
        realids[id] = i;
        pts[i].x = readlds();
        pts[i].y = readlds();
    }
    ve<ve<int> > g(n);
    int m;
    cin >> m;
    ve<pii> es;
    for(int i = 0; i < m; i++){
        int a, b;
        cin >> a >> b;
        a = realids[a]; b = realids[b];
        es.push_back(minmax(a, b));
    }
    sort(es.begin(), es.end());
    es.erase(unique(es.begin(), es.end()), es.end());
    for(auto [a, b] : es){
        g[a].push_back(b);
        g[b].push_back(a);
    }
        
    ve<ve<int> > faces = find_faces(pts, g, 0);
    delete_inner(pts, faces);
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
    ve<int> cnt(faces.size(), 0);
    scanline(pts, faces, cnt, users);
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
        if (done){ //this part sucks too
            ve<ve<int>> comps(z);
            for(int i = 0; i < z; ++i)
                comps[d.get(i)].push_back(i);
            ve<ve<int> > ans;
            for(auto i : comps){
                if(i.empty()) continue;
                ve<ve<int> > curadj(n);
                auto add_edge = [&](int u, int v){
                    curadj[u].push_back(v);
                    curadj[v].push_back(u);
                };
                for(auto k : i){
                    add_edge(faces[k].front(), faces[k].back());
                    for(int it = 1; it < faces[k].size(); it++)
                        add_edge(faces[k][it], faces[k][it - 1]);
                }
                ve<ve<int> > cur = find_faces(pts, curadj, 1);
                ans.push_back(cur[0]);
            }
            cout << ans.size() << "\n";
            for(auto i : ans){
                for(auto j : i)
                    cout << inputids[j] << " ";
                cout << "\n";
            }
            break;
        }
//        cout << iter++ << ": " << mn << ", " << mx << "\n";
    }
//    cout << "yay\n";
    return 0;
}
