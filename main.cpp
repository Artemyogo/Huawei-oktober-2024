#include <bits/stdc++.h>

#define ve vector
#define ll long long
#define fi first
#define se second

using namespace std;

using pii = pair<int, int>;

const int MAXN = 1e6+10;
#define M_PI 3.14159265358979323846

ll sqrt128(__int128 x){
    ll l = 0, r = 1e18;
    while(l < r){
        __int128 mid = (l + r) >> 1;
        if(mid*mid >= x)
            r = mid;
        else l = mid + 1;
    }
    return l;
}

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
    ll len() const{
        return sqrt128(__int128(x)*x + __int128(y)*y);
    }
    ll dst(const Point p) const{
        return (*this - p).len();
    }
};

struct timetracker {
    chrono::milliseconds timetaken;
    std::chrono::steady_clock::time_point begin;
    void begintimer() {
        begin = chrono::steady_clock::now();
    }
    int get() {
        return chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count();
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

inline bool operator <(const event& e1, const event& e2){
    if(!eq(e1.x, e2.x))
        return e1.x < e2.x;
    if(abs(e1.type) != abs(e2.type))
        return abs(e1.type) < abs(e2.type);
    else if(e1.type != e2.type)
        return e1.type < e2.type;
    else return edge_cmp(e1.e, e2.e);
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
        if(adj[u].empty()) continue;
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
            vector<int> cface;
            int v = i;
            int e = edge_id;
            ve<ve<int> > cfaces;
            map<int, bool> vis;
            while (!used[v][e]) {
                used[v][e] = true;
                cface.push_back(v);
                if(vis[v]){
                    cfaces.emplace_back();
                    do{
                        cfaces.back().push_back(cface.back());
                        cface.pop_back();
                    } while(cface.back() != v);
                }
                vis[v] = 1;
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
            reverse(cface.begin(), cface.end());
            cfaces.push_back(cface);
            for(auto face : cfaces){
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
    }
    return faces;
}

void delete_inner(const ve<Point>& p, ve<ve<int> >& faces){
    int n = p.size();
    ve<event> es;
    dsu ds(faces.size());
    ve<pair<pii, pii> > ces;
    for(int it = 0; it < faces.size(); it++){
        auto &f = faces[it];
        ces.push_back({minmax(f.front(), f.back()), {it, f.front() < f.back()}});
        for(int i = 1; i < f.size(); i++)
            ces.push_back({minmax(f[i], f[i - 1]), {it, f[i] < f[i - 1]}});
    }
    sort(ces.begin(), ces.end());
    for(int it = 0; it < ces.size(); it++){
        if((it == 0 || ces[it - 1].fi != ces[it].fi) && (it == ces.size() - 1 || ces[it + 1].fi != ces[it].fi)){
            auto [i, prev] = ces[it].fi;
            if(!ces[it].se.se)
                swap(prev, i);
            if(p[prev].x < p[i].x){
                es.push_back({1, p[prev].x, edge(p[prev], p[i], ces[it].se.fi + 1)});
                es.push_back({-1, p[i].x, edge(p[prev], p[i], ces[it].se.fi + 1)});
            }
            else if(p[prev].x > p[i].x){
                es.push_back({-1, p[prev].x, edge(p[i], p[prev], -ces[it].se.fi - 1)});
                es.push_back({1, p[i].x, edge( p[i], p[prev], -ces[it].se.fi - 1)});
            }
        }
        else if(it == ces.size() - 1 || ces[it + 1].fi != ces[it].fi){
            ds.unite(ces[it].se.fi, ces[it - 1].se.fi);
        }
    }
    sort(es.begin(), es.end());
    ve<bool> del(faces.size());
    set<edge, decltype(*edge_cmp)> st(edge_cmp);
    for(auto [tp, x, e] : es){
        if(tp == 1){
            if(del[ds.get(abs(e.up) - 1)]) continue;
            auto it = st.insert(e).first;
            if(it == st.begin()) continue;
            auto prv = prev(it);
            while(prv != st.begin() && del[ds.get(abs(prv->up) - 1)]){
                prv = prev(st.erase(prv));
            }
            if(it->up > 0 && !del[ds.get(abs(prv->up) - 1)] && prv->up > 0){
                del[ds.get(abs(it->up) - 1)] = 1;
            }
        }
        else if(tp == -1)
            if(st.find(e) != st.end())
                st.erase(e);
    }
    ve<ve<int> > nfaces;
    for(int i = 0; i < faces.size(); i++)
        if(!del[ds.get(i)])
            nfaces.push_back(faces[i]);
    faces.swap(nfaces);

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
int inputuserids[MAXN];


void scanline(const ve<Point>& p, ve<ve<int> >& faces, ve<int>& cnt, ve<Point>& users, ve<ve<int> >& who){
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
    for(int it = 0; it < users.size(); it++){
        auto &i = users[it];
        es.push_back({0, i.x, edge(i, i, it)});
    }
    sort(es.begin(), es.end());
    set<edge, decltype(*edge_cmp)> st(edge_cmp);
    for(int i = 0; i < es.size(); i++){
        auto [t, x, e] = es[i];
        if(t == 1){
            int prevsz = st.size();
            st.insert(e);
            assert(st.size() == prevsz + 1);
        }
        else if(t == -1){
            st.erase(e);
        }
        else{
            auto it = st.upper_bound(e);
            assert(it != st.begin());
            it--;
            cnt[it->up]++;
            who[it->up].push_back(e.up);
        }
    }
}

vector<pii> init_edges(vector<vector<int>> &faces, ve<int>& border) {
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
        else border.push_back(a[0]);
    }
    sort(border.begin(), border.end());
    border.erase(unique(border.begin(), border.end()), border.end());
    sort(edges.begin(), edges.end());
    edges.erase(unique(edges.begin(), edges.end()), edges.end());
    return edges;
}

const double EARTH_RADIUS = 6700.0; // Radius of Earth in kilometers

// Function to convert latitude and longitude to 3D coordinates
void latLonToXYZ(double latitude, double longitude, double& x, double& y, double& z) {
    // Convert degrees to radians
    double latRad = latitude * M_PI / 180.0;
    double lonRad = longitude * M_PI / 180.0;

    // Compute the 3D coordinates
    x = EARTH_RADIUS * cos(latRad) * cos(lonRad);
    y = EARTH_RADIUS * cos(latRad) * sin(lonRad);
    z = EARTH_RADIUS * sin(latRad);
}

double eval(vector<vector<int>> &ans, vector<Point> &pts) {
    double res = 0;
    for (auto &ve : ans) {
        double sum = 0;
        for (int i = 0; i < ve.size(); ++i) {
            int j = (i + 1) % ve.size();
            int k = (i + 2) % ve.size();
            Point A = pts[i], B = pts[j], C = pts[k];
            array<double, 3> a, b, c;
            latLonToXYZ(A.x / 1e15, A.y / 1e15, a[0], a[1], a[2]);
            latLonToXYZ(B.x / 1e15, B.y / 1e15, b[0], b[1], b[2]);
            latLonToXYZ(C.x / 1e15, C.y / 1e15, c[0], c[1], c[2]);
            array<double, 3> ab = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
            array<double, 3> bc = {c[0] - b[0], c[1] - b[1], c[2] - b[2]};
            double ab_l = sqrt(ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2]);
            double bc_l = sqrt(bc[0] * bc[0] + bc[1] * bc[1] + bc[2] * bc[2]);
            double cr = ab[0] * bc[0] + ab[1] * bc[1] + ab[2] * bc[2];
            sum += acos(cr / ab_l / bc_l);
        }
        res = max(res, sum);
    }
    return res;
}

mt19937 rng(5);


const int L = 512, R = 1024;

int TL = 9000;

Point center(ve<int>& v, ve<Point>& pts){
    double x = 0, y = 0, p = 0;
    int n = v.size();
    for(int i = 0; i < n; i++){
        Point a = pts[i];
        Point b = pts[(i + 1) % n];
        Point c = a - b;
        double len = sqrtl(double(c.x)*c.x + double(c.y)*c.y);
        x += len*(double(a.x+b.x)/2.);
        y += len*(double(a.y + b.y)/2.);
        p += len;
    }
    x /= p;
    y /= p;
    return Point(x, y);
}

int main(){
  //     assert(freopen("small_Minsk.txt", "r", stdin));
    timetracker timer;
    timer.begintimer();
    int n;
    cin >> n;
    if (n >= 1e5) TL = 28000;
    ve<Point> pts(n);
    for(int i = 0; i < n; i++){
        int id;
        cin >> id;
        inputids[i] = id;
        realids[id] = i;
        pts[i].x = readlds();
        pts[i].y = readlds();
    }
    ve<ve<int> > node_graph(n);
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
        node_graph[a].push_back(b);
        node_graph[b].push_back(a);
    }

    ve<ve<int> > faces = find_faces(pts, node_graph, 0);
    delete_inner(pts, faces);
    int z = faces.size();
    ve<Point> centers(z);
    for(int i = 0; i < z; i++)
        centers[i] = center(faces[i], pts);

    int t;
    cin >> t;
    ve<Point> users(t);

    for(int it = 0; it < t; it++){
        auto &i = users[it];
        int id;
        cin >> id;
        inputuserids[it] = id;
        i.x = readlds();
        i.y = readlds();
    }

    ve<int> cnt(faces.size(), 0);
    ve<ve<int> > who(faces.size());
    scanline(pts, faces, cnt, users, who);
    ve<int> border;
    auto edges = init_edges(faces, border);
    dsu d(z);

    ve<ve<int> > faceg(z);
    for(auto [a, b] : edges){
        faceg[a].push_back(b);
        faceg[b].push_back(a);
    }

    ve<int> borderdst(z, 1e9);
    {
        queue<int> q;
        for(auto i : border){
            q.push(i);
            borderdst[i] = 0;
        }
        while(!q.empty()){
            int v = q.front();
            q.pop();
            for(auto to : faceg[v])
                if(borderdst[to] > borderdst[v] + 1){
                    borderdst[to] = borderdst[v] + 1;
                    q.push(to);
                }
        }
    }
    ve<int> compall(z, -1);

    auto dfs = [&](int v){
        stack<int> st;
        st.push(v);
        while(!st.empty()){
            v = st.top();
            st.pop();
            for(auto to : faceg[v])
                if(compall[to] == -1){
                    compall[to] = compall[v];
                    st.push(to);
                }
        }
    };
    for(int i = 0; i < z; i++)
        if(compall[i] == -1){
            compall[i] = i;
            dfs(i);
        }

    vector<vector<int>> _ans;
    vector<vector<int>> _ansusers;
    double bst = 1e18;

    while (timer.get() < TL) { // this part sucks
        for(auto& i : faceg)
            shuffle(i.begin(), i.end(), rng);
        auto compare = [&](int a, int b){
            return borderdst[a] > borderdst[b];
        };
        priority_queue<int, vector<int>, decltype(compare)> start(compare);
        ve<bool> viscompall(z, 0);
        ve<int> startorder(z);
        iota(startorder.begin(), startorder.end(), 0);
        shuffle(startorder.begin(), startorder.end(), rng);
        for(auto i : startorder) {
            if(borderdst[i] == 0 && !viscompall[compall[i]]){
                viscompall[compall[i]] = 1;
                start.push(i);
            }
        }
        ve<int> compcol(z, -1), compsz = cnt;
        while(!start.empty()){
            int s = start.top();
            start.pop();
            if(compcol[s] != -1) continue;
            compcol[s] = s;
            auto comp = [&](int a, int b){
                return centers[s].dst(centers[a]) > centers[s].dst(centers[b]);
            };
            priority_queue<int, ve<int>, decltype(comp)> q(comp);
            q.push(s);
            while(!q.empty()){
                int v = q.top();
                q.pop();
                for(auto to : faceg[v]){
                    if (compcol[to] != -1) continue;
                    if (compsz[s] <= L && compsz[s] + compsz[to] <= R) {
                        compcol[to] = s;
                        compsz[s] += compsz[to];
                        q.push(to);
                    } else {
                        start.push(to);
                    }
                }

            }
        }
        bool done = true;

        vector<vector<int>> comp(z);
        for (int i = 0; i < z; ++i) {
            comp[compcol[i]].push_back(i);
        }
        vector<int> mrk(z, 1);
        vector<int> to_process;
        for (int i = 0; i < z; ++i) {
            assert(compsz[compcol[i]] <= R);
            if (comp[i].size() > 0 && compsz[i] < L) {
                to_process.push_back(i);
                for (auto &u : comp[i]) {
                    mrk[u] = 0;
                }
            }
        }

        vector<int> bdst(z, -1);
        vector<vector<int>> temp_g(z);
        vector<int> pt(z);
        shuffle(to_process.begin(), to_process.end(), rng);
        for (auto &ci : to_process) {
            auto cmp = comp[ci];
            vector<int> nei, bfs;
            for (auto &u : cmp) {
                for (auto &v : faceg[u]) {
                    if (mrk[v]) {
                        nei.push_back(v);
                        if (!~bdst[v]) {
                            bdst[v] = 0;
                            bfs.push_back(v);
                        }
                    }
                    temp_g[v].push_back(u);
                    temp_g[u].push_back(v);
                }
            }

            for (int pt = 0; pt < bfs.size(); ++pt) {
                int u = bfs[pt];
                for (auto &v : temp_g[u]) {
                    if (!~bdst[v]) {
                        bdst[v] = bdst[u] + 1;
                        bfs.push_back(v);
                    }
                }
            }

            sort(nei.begin(), nei.end());
            nei.erase(unique(nei.begin(), nei.end()), nei.end());

            if (rng() & 1) {
                for (auto &u : cmp) {
                    sort(temp_g[u].begin(), temp_g[u].end(), [&](int x, int y) {
                        return cnt[x] < cnt[y];
                    });
                }
                for (auto &u : nei) {
                    sort(temp_g[u].begin(), temp_g[u].end(), [&](int x, int y) {
                        return cnt[x] < cnt[y];
                    });
                }
            } else {
                for (auto &u : cmp) {
                    shuffle(temp_g[u].begin(), temp_g[u].end(), rng);
                }
                for (auto &u : nei) {
                    shuffle(temp_g[u].begin(), temp_g[u].end(), rng);
                }
            }

            auto compare = [&](int u, int v) {
                return bdst[u] > bdst[v];
            };
            priority_queue<int, vector<int>, decltype(compare)> s(compare);
            for (auto &u : nei) {
                s.push(u);
            }
            while (!s.empty()) {
                int u = s.top(); s.pop();
                int id = compcol[u];
                for (; pt[u] < temp_g[u].size(); ++pt[u]) {
                    int v = temp_g[u][pt[u]];
                    if (!mrk[v] && compsz[id] + cnt[v] <= R) {
                        mrk[v] = 1;
                        compcol[v] = id;
                        compsz[id] += cnt[v];
                        comp[id].push_back(v);
                        s.push(v);
                        break;
                    }
                }
                if (pt[u] != temp_g[u].size()) {
                    s.push(u);
                }
            }

            for (auto &u : nei) {
                temp_g[u].clear();
                pt[u] = 0;
                bdst[u] = -1;
            }
            for (auto &u: cmp) {
                temp_g[u].clear();
                pt[u] = 0;
                bdst[u] = -1;
            }
        }

        for (int i = 0; i < z; ++i) {
            if (compsz[compcol[i]] < L) {
                done = false;
            }
        }

        if (done){
            ve<ve<int> > comps(z);
            for(int i = 0; i < z; ++i)
                comps[compcol[i]].push_back(i);
            ve<ve<int> > ans;
            ve<ve<int> > ansusers;
            for(auto i : comps){
                if(i.empty()) continue;
                ve<ve<int> > curadj(n);
                ansusers.emplace_back();
                map<pii, bool> visedge;
                auto add_edge = [&](int u, int v){
                    if(visedge[{u, v}]) return;
                    curadj[u].push_back(v);
                    curadj[v].push_back(u);
                    visedge[{u, v}] = visedge[{v, u}] = 1;
                };
                for(auto k : i){
                    ansusers.back().insert(ansusers.back().end(), who[k].begin(), who[k].end());
                    add_edge(faces[k].front(), faces[k].back());
                    for(int it = 1; it < faces[k].size(); it++)
                        add_edge(faces[k][it], faces[k][it - 1]);
                }
                ve<ve<int> > cur = find_faces(pts, curadj, 1);
                ans.push_back(cur[0]);
            }

            double val = eval(ans, pts);

            if (val < bst) {
                bst = val;
                _ans = ans;
                _ansusers = ansusers;
            }

        }
    }
    cout << _ans.size() << "\n";
    int it = 0;
    for(auto i : _ans){
        cout << i.size() << "\n";
        for(auto j : i)
            cout << inputids[j] << " ";
        cout << "\n";
        cout << _ansusers[it].size() << "\n";
        for(auto j : _ansusers[it])
            cout << inputuserids[j] << " ";
        cout << "\n";
        it++;
    }
    cerr << bst;
    return 0;
}
