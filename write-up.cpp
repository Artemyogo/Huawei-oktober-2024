/// Gist of the solution: 
/// 1) Convert graph into a "usable format" - find all faces, delete redundant faces (those that lie inside of another face), 
///     find the number of users inside of each face
/// 2) Start the process of finding possible solutions while we still have time. This process consists of 2 parts: 
///    a) Split faces into rough components of a size around 512
///    b) For every small component (its size is less than 512), try to distribute its faces between this componenent's neighbors
///    c) Verify that resulting solution fits the requirments and update the answer if needed

#include <bits/stdc++.h>

using namespace std;

/// Defines that will be latter used to shorten the code

#define ve vector
#define ll long long
#define fi first
#define se second

using pii = pair<int, int>;

/// Max size of N used for static arrays
const int MAXN = 1e6+10;
/// Value of Pi
#define M_PI 3.14159265358979323846
/// Constraints on the number of users in one face
const int L = 512, R = 1024;

/// Random Variable with a constant seed
mt19937 rng(5);

/// Timer
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

/// Reading floating point number and converting it into a 64-bit integer
ll readlds(){
    string st;
    cin >> st;
    size_t pos = st.find(".");
    if(pos == -1)
        pos = st.size();
    else st.erase(pos, 1);
    while(st.size() - pos < 16)
        st.push_back('0');
    return stoll(st);
}

/// Point struct that will be used for all computations
/// Since there were only 15 digits after floating point, we used 64-bit integer data type for maximum precision
/// Multiplication is done in 128-bit integers
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


/// Since we initially used floating point calculations, instead of redefining functions
/// we just used macros
#define le(a, b) (a <= b)
#define eq(a, b) (a == b)

/// Function for finding a sign of a number
int sgn(const __int128& x) { return eq(x, 0) ? 0 : le(x, 0) ? -1 : 1; }

// Struct of edge of polygon where up is id of face/user
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

//struct for event in scanline sorted by x
struct event{
    int type; //-1 — del, 0 — get, 1 - add
    ll x;
    edge e;
};

//comparator for non-intersecting edges where e1 < e2 if e1 is lower than e2
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

// comparator for event we need to add and del first and only then get
inline bool operator <(const event& e1, const event& e2){
    if(!eq(e1.x, e2.x))
        return e1.x < e2.x;
    if(abs(e1.type) != abs(e2.type))
        return abs(e1.type) < abs(e2.type);
    else if(e1.type != e2.type)
        return e1.type < e2.type;
    else return edge_cmp(e1.e, e2.e);
}


/// Union-find data structure with path-compression and small-to-large merging
/// For an easier implementation, we used a version of union-find that utilizes one array p
/// if p[u] >= 0, them p[u] is a parent of u
/// if p[u] < 0, then it means that u is a root of its component and p[u] is a negative size of the component
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

//number of innner faces since this number is used only while finding outerfaces
int cntinner;

vector<vector<int>> find_faces(vector<Point> vertices, vector<vector<int>> adj, bool inner) {
    cntinner = 0;
    int n = vertices.size();
    vector<vector<bool>> used(n);

    /// Here we are deleting all points that are not part of any cycle and therefore can't be part of a face
    /// We are doing it in a bfs-like manner, where we add vertices that we need to delete in a queue (though we used vector instead of a queue)
    vector<int> bfs;
    for (int i = 0; i < n; ++i) {
        /// adj[i].size() == 1 means that vertex has only one neighbor, and therefore it should be deleted
        if (adj[i].size() == 1) {
            bfs.push_back(i);
        }
    }
    for (int pt = 0; pt < bfs.size(); ++pt) {
        int u = bfs[pt];
        if(adj[u].empty()) continue;
        int v = adj[u][0]; /// v is the only adjacent vertex to u
        adj[u].clear();

        /// deleting u from the adjacency-list of v
        for (auto &x : adj[v]) {
            if (x == u) {
                swap(adj[v].back(), x);
                adj[v].pop_back();
                break;
            }
        }
        /// adding v to the queue if it now becomes a vertex with an only one neighbor
        if (adj[v].size() == 1) {
            bfs.push_back(v);
        }
    }


    for (int i = 0; i < n; i++) {
        used[i].assign(adj[i].size(), 0);
        //comparator for angle of edge
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
            vector<int> cface;
            int v = i;
            int e = edge_id;
            ve<ve<int> > cfaces;
            map<int, bool> vis;
            //while cycle is not completed
            while (!used[v][e]) {
                used[v][e] = true;
                cface.push_back(v);
                //distinguishing cycles in cycles
                if(vis[v]){
                    cfaces.emplace_back();
                    do{
                        cfaces.back().push_back(cface.back());
                        cface.pop_back();
                    } while(cface.back() != v);
                }
                vis[v] = 1;
                int u = adj[v][e];
                //the same comparator as before
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
            //finding if face is inner or outer by signed sum
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
                else cntinner++;
            }
        }
    }
    return faces;
}

//deleting faces in faces using scanline
//if there is lower edge of the face next below to the lower edge of the second face than second face needs to be deleted
void delete_inner(const ve<Point>& p, ve<ve<int> >& faces){
    int n = p.size();
    ve<event> es;
    dsu ds(faces.size());
    ve<pair<pii, pii> > ces;
    //memorising edges with structure {{PointID, PointID}, {FaceID, IsUpper}}
    for(int it = 0; it < faces.size(); it++){
        auto &f = faces[it];
        ces.push_back({minmax(f.front(), f.back()), {it, f.front() < f.back()}});
        for(int i = 1; i < f.size(); i++)
            ces.push_back({minmax(f[i], f[i - 1]), {it, f[i] < f[i - 1]}});
    }
    //sorting events
    sort(ces.begin(), ces.end());
    
    //adding outer edges to events
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
    
    //sorting events and memorising which faces need to be deleted
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


/// For a convenience we mapped every point id to a number from 0 to n and every user id to a number from 0 to t
/// realids maps point id to number
/// inputids maps number to point id
/// inputuserids maps number to user id
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

/// Function that finds edges between adjacent faces and also array of faces that lie on the border
/// Since every edge in the planar graph has only two faces (one for each side) that use this edge
/// So, for every edge we find two faces (or possible only one if the edge is on the border) lying on both sides of the edge, 
///  and add an edge between these two faces (or, add the face to the "border" list)
vector<pii> init_edges(vector<vector<int>> &faces, ve<int>& border) {
    map<pii, vector<int>> mp;
    for (int i = 0; i < faces.size(); ++i) {
        auto &ve = faces[i];
        assert(ve.size() > 1);
        for (int j = 0; j < ve.size(); ++j) {
            int u = ve[j], v = ve[(j + 1) % ve.size()];
            /// minmax(u, v) = {min(u, v), max(u, v)}
            /// it is done to handle the cases in which vertices u v are in different order
            mp[minmax(u, v)].push_back(i);
        }
    }
    vector<pii> edges;
    for (auto &[_, a] : mp) {
        sort(a.begin(), a.end());
        a.erase(unique(a.begin(), a.end()), a.end());
        /// if a.size() > 1, that means that faces a[0] and a[1] are connected by this edge
        if (a.size() > 1) {
            edges.emplace_back(a[0], a[1]);
        } /// if a.size() == 1, that means that face lies on the border
        else border.push_back(a[0]);
    }
    /// getting rid of duplicates 
    sort(border.begin(), border.end());
    border.erase(unique(border.begin(), border.end()), border.end());
    sort(edges.begin(), edges.end());
    edges.erase(unique(edges.begin(), edges.end()), edges.end());
    return edges;
}

const double EARTH_RADIUS = 6700.0; /// Radius of Earth in kilometers

/// Function to convert latitude and longitude to 3D coordinates
void latLonToXYZ(double latitude, double longitude, double& x, double& y, double& z) {
    /// Convert degrees to radians
    double latRad = latitude * M_PI / 180.0;
    double lonRad = longitude * M_PI / 180.0;
    /// Compute the 3D coordinates using spherical coordinates formula
    x = EARTH_RADIUS * cos(latRad) * cos(lonRad);
    y = EARTH_RADIUS * cos(latRad) * sin(lonRad);
    z = EARTH_RADIUS * sin(latRad);
}

/// Function that finds an agle made by three points
double eval(Point A, Point B, Point C){
    /// Converting points A, B, C to floating-point coordinates
    array<double, 3> a, b, c;
    latLonToXYZ(A.x / 1e15, A.y / 1e15, a[0], a[1], a[2]);
    latLonToXYZ(B.x / 1e15, B.y / 1e15, b[0], b[1], b[2]);
    latLonToXYZ(C.x / 1e15, C.y / 1e15, c[0], c[1], c[2]);

    /// Finding vectors AB and BC
    array<double, 3> ab = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    array<double, 3> bc = {c[0] - b[0], c[1] - b[1], c[2] - b[2]};
    /// Lengths of vectors AB and BC
    double ab_l = sqrt(ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2]);
    double bc_l = sqrt(bc[0] * bc[0] + bc[1] * bc[1] + bc[2] * bc[2]);
    /// Cross product of AB and BC
    double cr = ab[0] * bc[0] + ab[1] * bc[1] + ab[2] * bc[2];
    return acos(cr / ab_l / bc_l);
}

/// Function that evaluates the score of an answer
double eval(vector<vector<int>> &ans, vector<Point> &pts) {
    double res = 0;
    for (auto &v : ans) {
        /// Total curvature of a face
        double sum = 0;
        for (int i = 0; i < v.size(); ++i) {
            int j = (i + 1) % v.size();
            int k = (i + 2) % v.size();
            Point A = pts[i], B = pts[j], C = pts[k];
            sum += eval(A, B, C);
        }
        /// Result is the maximal curvature
        res = max(res, sum);
    }
    return res;
}

struct pqnode{
    double ang;
    int cnt;
    int id;
};

inline bool operator <(const pqnode& a, const pqnode& b){
    return a.ang * b.cnt == b.ang * a.cnt ? a.id < b.id : a.ang * b.cnt < b.ang * a.cnt;
}

int main(){
    ios_base::sync_with_stdio(0); cin.tie(0);
    timetracker timer;
    /// Starting the timer
    timer.begintimer();
    int n;
    cin >> n;

    /// Adjusting TL depending on a size of n
    int TL = 9000;
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
    /// Getting rid of duplicate edges
    sort(es.begin(), es.end());
    es.erase(unique(es.begin(), es.end()), es.end());
    for(auto [a, b] : es){
        node_graph[a].push_back(b);
        node_graph[b].push_back(a);
    }

    /// Finding faces
    ve<ve<int> > faces = find_faces(pts, node_graph, 0);
    /// Deleting redundant faces
    delete_inner(pts, faces);
    
    int z = faces.size();

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

    /// cnt[i] - number of users inside of the i'th face
    /// who[i] - list of users that are inside of the i'th face
    ve<int> cnt(faces.size(), 0);
    ve<ve<int> > who(faces.size());
    /// Finding the number of users inside of each face using scanline
    scanline(pts, faces, cnt, users, who);
    /// Border - list of faces that lie on the border of the graph
    ve<int> border;
    /// Finding edges between adjacent faces
    auto edges = init_edges(faces, border);

    /// faceg - graph made by faces and edges between these faces
    ve<ve<int> > faceg(z);
    for(auto [a, b] : edges){
        faceg[a].push_back(b);
        faceg[b].push_back(a);
    }

    /// Part of our greedy strategy requires distance from the border
    /// We are determining it using bfs
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

    /// Finding connnected components of the graph using dfs
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
    for(int i = 0; i < z; i++) {
        if(compall[i] == -1){
            compall[i] = i;
            dfs(i);
        }
    }

    /// _ans - list of final faces
    /// _ansusers - lists of users lying inside of the corresponding faces
    /// bst - score of the best solution
    vector<vector<int>> _ans;
    vector<vector<int>> _ansusers;
    double bst = 1e18;

    /// Repeat the process while we have time
    while (timer.get() < TL) {
        /// Shuffling adjacency lists of faces to have ~random solutions 
        for(auto& i : faceg)
            shuffle(i.begin(), i.end(), rng);
        /// In the first step of the algorithm, we are splitting the vertices in the components of the size around 512.
        /// We're going to do that using greedy strategy: 
        /// First, choose a face that is the closest to the border (we're doing that to distribute faces more evenly)
        ///  and to avoid cases where we will have some faces "cut off" the whole graph)
        /// Then, start adding new faces to it in a dijikstra-like manner, comparing faces using a greedy criteria described near the pqnode.
        /// When we can't add a face to our current component, add it to the list of possible new starts
        /// Repeat until the list of starts is non empty

        /// Comparator for starts
        auto compare = [&](int a, int b){
            return borderdst[a] > borderdst[b];
        };
        priority_queue<int, vector<int>, decltype(compare)> start(compare);
        ve<bool> viscompall(z, 0);
        ve<int> startorder(z);
        iota(startorder.begin(), startorder.end(), 0);
        shuffle(startorder.begin(), startorder.end(), rng);
        for(auto i : startorder) {
            /// In each connected component, initially we want to have only one start
            if(borderdst[i] == 0 && !viscompall[compall[i]]){
                viscompall[compall[i]] = 1;
                start.push(i);
            }
        }

        /// compcol[i] - components to which the i'th face belongs
        /// compsz[i] - number of users inside of the i'th component
        ve<int> compcol(z, -1), compsz = cnt;
        vector<bool> used(z);
        /// vals[i] - current value of the i'th face (we're comparing them inside of the heap)
        ve<pqnode> vals(z);
        while(!start.empty()){
            int s = start.top();
            start.pop();
            if(compcol[s] != -1) continue; /// This start is already a part of some component
            compcol[s] = s;
            priority_queue<pqnode> q;
            /// outer is a set with edges on the border of our current component
            set<pii> outer;
            vals[s] = {-1, 1, s};
            q.push(vals[s]);
            while(!q.empty()){
                auto [cang, ccnt, v] = q.top();
                q.pop();
                if(used[v]) continue;
                used[v] = 1;
                int prev = faces[v].back();
                for(auto i : faces[v]) {
                    if(outer.find(minmax(prev, i)) != outer.end()) {
                        outer.erase(minmax(prev, i));
                    }
                    else{
                        outer.insert(minmax(prev, i));
                    }
                    prev = i;
                }
                if(v != s){
                    if(compsz[s] >= L || compsz[s] + compsz[v] > R){
                        used[v] = 0;
                        start.push(v);
                        continue;
                    }
                    compsz[s] += compsz[v];
                    compcol[v] = s;
                }
                for(auto to : faceg[v]){
                    if (compcol[to] != -1) continue;
                    auto &f = faces[to];
                    vals[to] = {0, 1, to};
                    for(int i = 0; i < f.size(); i++){
                        int a = f[i];
                        int b = f[(i + 1) % f.size()];
                        int c = f[(i + 2) % f.size()];
                        if(outer.find(minmax(a, b)) != outer.end() && outer.find(minmax(b, c)) != outer.end()){
                            vals[to].cnt++;
                            vals[to].ang += eval(pts[a], pts[b], pts[c]);
                        }
                    }
                    q.push(vals[to]);
                }
            }
        }


        /// Here starts the part where we split small components back into faces and distribute them between neighbors of the component
        /// Some definitions: 
        /// bad component - component that has less than 512 users
        /// good component - component that is not bad
        /// bad/good face - faces that is a part of bad/good component
        
        bool done = true;

        /// comp[i] - list of faces lying in the i'th component
        vector<vector<int>> comp(z);
        for (int i = 0; i < z; ++i) {
            comp[compcol[i]].push_back(i);
        }
        
        /// mrk[i] - marks whether the i'th face is part of a good component
        vector<int> mrk(z, 1);
        /// to_process - list of all small components that we need to break down 
        vector<int> to_process;
        for (int i = 0; i < z; ++i) {
            if (comp[i].size() > 0 && compsz[i] < L) {
                to_process.push_back(i);
                for (auto &u : comp[i]) {
                    mrk[u] = 0;
                }
            }
        }

        /// temp_g - local array that we will use to keep a graph on only relevant faces
        vector<vector<int>> temp_g(z);
        vector<int> pt(z);
        shuffle(to_process.begin(), to_process.end(), rng);
        for (auto &ci : to_process) {
            auto cmp = comp[ci];
            /// nei - list of neighboring good faces
            vector<int> nei;
            /// same as in the first part
            set<pii> outer;
            auto toggle = [&](pii e) {
                if (outer.count(e)) {
                    outer.erase(e);
                } else {
                    outer.insert(e);
                }
            };
            for (auto &u : cmp) {
                int prev = faces[u].back();
                for(auto i : faces[u]) {
                    toggle(minmax(prev, i));
                    prev = i;
                }
                for (auto &v : faceg[u]) {
                    if (mrk[v]) { /// v is a part of a good component
                        nei.push_back(v);
                    }
                    temp_g[v].push_back(u);
                    temp_g[u].push_back(v);
                }
            }

            sort(nei.begin(), nei.end());
            nei.erase(unique(nei.begin(), nei.end()), nei.end());

            /// We randomly decide between shuffling faces or sorting them by the number of users inside
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

            /// Here we used a set because it turned out to be faster than priority_queue
            set<pqnode> s;
            /// Function that recalcs a value of u-th face
            auto recalc = [&](int u) {
                auto &f = faces[u];
                vals[u] = {0, 1, u};
                for (int i = 0; i < f.size(); ++i) {
                    int a = f[i];
                    int b = f[(i + 1) % f.size()];
                    int c = f[(i + 2) % f.size()];
                    if(outer.count(minmax(a, b)) && outer.count(minmax(b, c))) {
                        vals[u].cnt++;
                        vals[u].ang += eval(pts[a], pts[b], pts[c]);
                    }
                }
            };

            /// In our set we keep only good faces, so initally we insert only them
            for (auto &u : nei) {
                recalc(u);
                s.insert(vals[u]);
            }

            /// The process looks like this: 
            /// 1) We take "the best" face from the set
            /// 2) We are trying to add the next face from its adjacency list to the component of our face
            /// 3) If we can do that, we recalc the value of all affected faces and return to the Step 1
            /// 4) We will reinsert our face to the set only if there are any other faces we can theoretically add to it
            while(!s.empty()) {
                auto [_, __, u] = *prev(s.end());
                s.erase(prev(s.end()));
                int id = compcol[u];
                for (; pt[u] < temp_g[u].size(); ++pt[u]) { 
                    int v = temp_g[u][pt[u]];
                    if (!mrk[v] && compsz[id] + cnt[v] <= R) {
                        mrk[v] = 1;
                        compcol[v] = id;
                        compsz[id] += cnt[v];
                        int prev = faces[v].back();
                        /// Update the outer face 
                        for(auto i : faces[v]) {
                            toggle(minmax(prev, i));
                            prev = i;
                        }
                        /// recalc the value of all possibly affected faces
                        recalc(v);
                        s.insert(vals[v]);
                        for (auto &to : temp_g[v]) {
                            if (s.count(vals[to])) {
                                s.erase(vals[to]);
                                recalc(to);
                                s.insert(vals[to]);
                            }
                        }
                        break;
                    }
                }
                /// Insert back only if there are any other faces that we can theorically connect to u
                if (pt[u] != temp_g[u].size()) {
                    recalc(u);
                    s.insert(vals[u]);
                }
            }

            /// Clear all local arrays we used
            for (auto &u : nei) {
                temp_g[u].clear();
                pt[u] = 0;
            }
            for (auto &u: cmp) {
                temp_g[u].clear();
                pt[u] = 0;
            }
        }

        /// If there is a component with a size less than L, then we failed and our solution shouldn't be further considered
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
                set<pii> visedge;
                auto add_edge = [&](int u, int v){
                    if(visedge.find({u, v}) != visedge.end() || visedge.find({v, u}) != visedge.end()) return;
                    curadj[u].push_back(v);
                    curadj[v].push_back(u);
                    visedge.insert({u, v});
                };
                for(auto k : i){
                    ansusers.back().insert(ansusers.back().end(), who[k].begin(), who[k].end());
                    add_edge(faces[k].front(), faces[k].back());
                    for(int it = 1; it < faces[k].size(); it++)
                        add_edge(faces[k][it], faces[k][it - 1]);
                }
                ve<ve<int> > cur = find_faces(pts, curadj, 1);
                done &= cur.size() == 1 && i.size() == cntinner;
                ans.push_back(cur[0]);
            }
            if(!done) continue;

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
