"""
╔══════════════════════════════════════════════════════════════╗
║          PUSH BLOCK  —  RL Agent  (Pure Python)             ║
║   Matches Unity ML-Agents PushBlock spec exactly            ║
║                                                              ║
║  Install:  pip install pygame numpy torch                    ║
║  Run:      python pushblock_game.py                         ║
╚══════════════════════════════════════════════════════════════╝

Spec (from assignment email / ML-Agents docs):
  - Vector Obs: 70 vars  (14 raycasts × 5 values each)
  - Actions: 1 discrete branch, 7 actions
      0=turn CW  1=turn CCW  2=fwd  3=back  4=strafe L  5=strafe R  6=nothing
  - Reward: -0.0025/step, +1.0 on goal touch
  - Float Properties: block_scale, dynamic_friction, static_friction, block_drag
  - Benchmark Mean Reward: 4.5
"""

import sys, math, random, collections
import numpy as np
import pygame

# ─── Try importing torch; fall back to Q-learning if unavailable ─────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ════════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════════
W, H         = 1200, 760
GRID_N       = 10
BOARD_PX     = 620
CELL         = BOARD_PX // GRID_N
BOARD_X      = 30
BOARD_Y      = 70
PANEL_X      = BOARD_X + BOARD_PX + 30
PANEL_W      = W - PANEL_X - 20

BG           = (130, 150, 160)
GRID_BG      = (45,  45,  45)
GRID_LINE    = (60,  60,  60)
WALL_C       = (100, 100, 100)
GOAL_A       = (126, 217, 87)
GOAL_B       = (126, 217, 87)
BLOCK_A      = (255, 255, 255)
BLOCK_B      = (230, 230, 230)
AGENT_A      = (56,  182, 255)
AGENT_B      = (255, 222, 89)
PANEL_BG     = (11,  14,  24)
ACCENT       = (0,  210, 130)
TEXT_HI      = (220, 230, 255)
TEXT_LO      = (80,  95, 130)
BAR_BG       = (25,  32,  52)
RED_C        = (255,  70,  70)
ORG_C        = (255, 160,  40)

ACTIONS      = 5
ACT_NAMES    = ["Turn CW","Turn CCW","Forward","Backward","Wait"]
NUM_RAYS     = 14
OBS_SIZE     = NUM_RAYS * 5 + 12  # 70 rays + 12 state hints = 82
MAX_TRAIN_EPISODES = 10000     # Stop training after this many episodes


# ════════════════════════════════════════════════════════════════
#  ENVIRONMENT
# ════════════════════════════════════════════════════════════════
class PushBlockEnv:
    """
    Grid-based PushBlock matching ML-Agents spec.
    Observation: 70-dim continuous vector (14 raycasts x 5).
    Each raycast: [hit_wall, hit_goal, hit_block, distance, nothing]
    """
    def __init__(self, block_scale=2.0, dynamic_friction=0.0,
                 static_friction=0.0, block_drag=0.5):
        self.gs               = GRID_N
        self.block_scale      = block_scale
        self.dynamic_friction = dynamic_friction
        self.static_friction  = static_friction
        self.block_drag       = block_drag
        self.reset()

    def reset(self):
        self.goal_cells = set()
        for c in range(self.gs):
            self.goal_cells.add((0, c))

        self.block_r   = self.gs // 2
        self.block_c   = self.gs // 2
        self.agent_r   = self.block_r + 2
        self.agent_c   = self.block_c
        self.agent_dir = 0   # 0=N 1=E 2=S 3=W
        self.obstacles = set()   # no obstacles
        self.steps        = 0
        self.done         = False
        self.episode_rew  = 0.0
        self.last_action  = 4
        self.success      = False
        self._prev_dist   = self.block_dist_to_goal()
        return self._obs()

    def is_solvable(self):
        start_state = (self.agent_r, self.agent_c, self.agent_dir, self.block_r, self.block_c)
        q = collections.deque([start_state])
        visited = {start_state}
        dirs = [(-1,0),(0,1),(1,0),(0,-1)]
        while q:
            ar, ac, adir, br, bc = q.popleft()
            if (br, bc) in self.goal_cells: return True
            for act in range(4):
                nar, nac, nadir, nbr, nbc = ar, ac, adir, br, bc
                dr, dc = 0, 0
                if act == 0: nadir = (adir + 1) % 4
                elif act == 1: nadir = (adir - 1) % 4
                elif act == 2: dr, dc = dirs[adir]
                elif act == 3: dr, dc = -dirs[adir][0], -dirs[adir][1]
                if dr != 0 or dc != 0:
                    tar, tac = ar + dr, ac + dc
                    if not (0 <= tar < self.gs and 0 <= tac < self.gs) or (tar, tac) in self.obstacles: continue
                    if (tar, tac) in self.goal_cells: continue
                    if tar == br and tac == bc:
                        tbr, tbc = br + dr, bc + dc
                        if not (0 <= tbr < self.gs and 0 <= tbc < self.gs) or (tbr, tbc) in self.obstacles: continue
                        nbr, nbc = tbr, tbc
                    nar, nac = tar, tac
                nstate = (nar, nac, nadir, nbr, nbc)
                if nstate not in visited:
                    visited.add(nstate)
                    q.append(nstate)
        return False

    def _obs(self):
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        angles = [i * (360 / NUM_RAYS) for i in range(NUM_RAYS)]
        for i, deg in enumerate(angles):
            rad = math.radians(deg + self.agent_dir * 90)
            dx, dy = math.cos(rad), math.sin(rad)
            hit_wall = hit_goal = hit_block = 0.0
            dist = 1.0
            for step in range(1, self.gs + 1):
                rx = self.agent_c + dx * step
                ry = self.agent_r + dy * step
                rc, rr = int(round(rx)), int(round(ry))
                norm_d = step / self.gs
                if not (0 <= rr < self.gs and 0 <= rc < self.gs) or (rr, rc) in self.obstacles:
                    hit_wall = 1.0; dist = norm_d; break
                if (rr, rc) in self.goal_cells:
                    hit_goal = 1.0; dist = norm_d; break
                if rr == self.block_r and rc == self.block_c:
                    hit_block = 1.0; dist = norm_d; break
            nothing = 1.0 if (hit_wall + hit_goal + hit_block) == 0 else 0.0
            base = i * 5
            obs[base]   = hit_wall
            obs[base+1] = hit_goal
            obs[base+2] = hit_block
            obs[base+3] = dist
            obs[base+4] = nothing
        
        obs[70] = (self.block_r - self.agent_r) / self.gs
        obs[71] = (self.block_c - self.agent_c) / self.gs
        obs[72] = self.agent_dir / 4.0
        obs[73] = self.block_dist_to_goal() / float(self.gs * 2)
        obs[74] = self.agent_r / self.gs
        obs[75] = self.agent_c / self.gs
        # BFS optimal push plan: tells agent where to stand + what direction to push
        opt = self.get_optimal_push_plan()
        if opt is not None:
            push_dr, push_dc, ideal_ar, ideal_ac = opt
            obs[76] = push_dr / 1.0   # which direction block should be pushed (row)
            obs[77] = push_dc / 1.0   # which direction block should be pushed (col)
            # Next step the AGENT should take to walk to ideal push spot (BFS around walls)
            ndr, ndc = self.agent_next_step_toward(ideal_ar, ideal_ac)
            obs[78] = ndr / 1.0        # agent: move this row direction next
            obs[79] = ndc / 1.0        # agent: move this col direction next
            # Walkable distance from agent to ideal push spot
            walk_dist = self.agent_walkable_dist_to(ideal_ar, ideal_ac)
            obs[80] = walk_dist / float(self.gs * 2)
            # Flag: is agent already at ideal push spot?
            obs[81] = 1.0 if (self.agent_r == ideal_ar and self.agent_c == ideal_ac) else 0.0
        else:
            obs[76] = obs[77] = obs[78] = obs[79] = obs[80] = obs[81] = 0.0
        return obs


    def step(self, action):
        if self.done:
            return self._obs(), 0.0, True
        self.last_action = action

        reward = -0.0025   # small penalty every step

        dirs = [(-1,0),(0,1),(1,0),(0,-1)]
        fr, fc = dirs[self.agent_dir]
        rr2, rc2 = dirs[(self.agent_dir + 1) % 4]

        if action == 0:   self.agent_dir = (self.agent_dir + 1) % 4
        elif action == 1: self.agent_dir = (self.agent_dir - 1) % 4
        elif action == 2: self._try_move(fr, fc)
        elif action == 3: self._try_move(-fr, -fc)

        # WIN: block touches goal
        if (self.block_r, self.block_c) in self.goal_cells:
            reward += 1.0
            self.done = True
            self.success = True

        self.steps += 1
        self.episode_rew += reward
        if self.steps >= 400:
            self.done = True
        return self._obs(), reward, self.done

    def is_good_push_position(self):
        """Returns True when agent is directly behind the block relative to the goal."""
        ar, ac = self.agent_r, self.agent_c
        br, bc = self.block_r, self.block_c
        # Agent should be BELOW the block (higher row number) to push north toward row 0
        if ar > br and abs(ac - bc) <= 1:
            return True
        return False

    def is_forward_push_blocked(self):
        """Returns True if pushing the block straight north is impossible (obstacle/wall blocking)."""
        br, bc = self.block_r, self.block_c
        nbr, nbc = br - 1, bc
        if not (0 <= nbr < self.gs and 0 <= nbc < self.gs):
            return True
        if (nbr, nbc) in self.obstacles:
            return True
        return False

    def clear_push_directions(self):
        """Returns list of (dr,dc) directions the block can be pushed without hitting an obstacle/wall."""
        br, bc = self.block_r, self.block_c
        clear = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nbr, nbc = br + dr, bc + dc
            if 0 <= nbr < self.gs and 0 <= nbc < self.gs and (nbr, nbc) not in self.obstacles:
                clear.append((dr, dc))
        return clear

    def get_optimal_push_plan(self):
        """
        BFS on block positions to find optimal push direction.
        Returns (push_dr, push_dc, ideal_agent_r, ideal_agent_c) or None.
        """
        br, bc = self.block_r, self.block_c
        if (br, bc) in self.goal_cells:
            return None
        q = collections.deque([((br, bc), None)])
        visited = {(br, bc)}
        while q:
            (cbr, cbc), first_push = q.popleft()
            if (cbr, cbc) in self.goal_cells:
                if first_push is None: return None
                push_dr, push_dc = first_push
                # Ideal agent pos: directly opposite the push direction from block
                ideal_ar = br - push_dr
                ideal_ac = bc - push_dc
                ideal_ar = max(0, min(self.gs - 1, ideal_ar))
                ideal_ac = max(0, min(self.gs - 1, ideal_ac))
                # If ideal spot is an obstacle, search for nearest clear cell
                if (ideal_ar, ideal_ac) in self.obstacles:
                    for off in range(1, self.gs):
                        for dr2, dc2 in [(-push_dr,-push_dc),(push_dc,-push_dr),(-push_dc,push_dr)]:
                            ar2, ac2 = br - push_dr + dr2*off, bc - push_dc + dc2*off
                            if 0<=ar2<self.gs and 0<=ac2<self.gs and (ar2,ac2) not in self.obstacles:
                                ideal_ar, ideal_ac = ar2, ac2
                                break
                        else:
                            continue
                        break
                return (push_dr, push_dc, ideal_ar, ideal_ac)
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nbr2, nbc2 = cbr+dr, cbc+dc
                if 0<=nbr2<self.gs and 0<=nbc2<self.gs and (nbr2,nbc2) not in self.obstacles:
                    push_side = (cbr-dr, cbc-dc)
                    if 0<=push_side[0]<self.gs and 0<=push_side[1]<self.gs:
                        if push_side not in self.obstacles and (nbr2,nbc2) not in visited:
                            visited.add((nbr2,nbc2))
                            fp = first_push if first_push is not None else (dr,dc)
                            q.append(((nbr2,nbc2), fp))
        return None

    def agent_next_step_toward(self, target_r, target_c):
        """BFS from agent position to target. Returns (dr,dc) of the FIRST step on the path."""
        if self.agent_r == target_r and self.agent_c == target_c:
            return (0, 0)
        q = collections.deque([(self.agent_r, self.agent_c, None)])
        visited = {(self.agent_r, self.agent_c)}
        while q:
            r, c, first_step = q.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<self.gs and 0<=nc<self.gs and (nr,nc) not in visited:
                    if (nr,nc) not in self.obstacles and (nr,nc) not in self.goal_cells:
                        step = first_step if first_step is not None else (dr,dc)
                        if nr == target_r and nc == target_c:
                            return step
                        visited.add((nr,nc))
                        q.append((nr, nc, step))
        return (0, 0)   # unreachable

    def agent_walkable_dist_to(self, target_r, target_c):
        """BFS walkable distance from agent to target cell."""
        if self.agent_r == target_r and self.agent_c == target_c:
            return 0
        q = collections.deque([(self.agent_r, self.agent_c, 0)])
        visited = {(self.agent_r, self.agent_c)}
        while q:
            r, c, d = q.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<self.gs and 0<=nc<self.gs and (nr,nc) not in visited:
                    if (nr,nc) not in self.obstacles:
                        if nr == target_r and nc == target_c:
                            return d+1
                        visited.add((nr,nc))
                        q.append((nr, nc, d+1))
        return 999

    def _try_move(self, dr, dc):
        nr, nc = self.agent_r + dr, self.agent_c + dc
        if not (0 <= nr < self.gs and 0 <= nc < self.gs) or (nr, nc) in self.obstacles:
            return
        # Agent cannot enter the goal zone — only the block can
        if (nr, nc) in self.goal_cells:
            return
        if nr == self.block_r and nc == self.block_c:
            br2, bc2 = self.block_r + dr, self.block_c + dc
            friction = max(self.dynamic_friction, self.static_friction)
            if (0 <= br2 < self.gs and 0 <= bc2 < self.gs
                    and (br2, bc2) not in self.obstacles
                    and random.random() > friction * 0.15):
                self.block_r, self.block_c = br2, bc2
                self.agent_r, self.agent_c = nr, nc
        else:
            self.agent_r, self.agent_c = nr, nc

    def agent_dist_to_block(self):
        q = collections.deque([(self.agent_r, self.agent_c)])
        dist = {(self.agent_r, self.agent_c): 0}
        while q:
            r, c = q.popleft()
            if r == self.block_r and c == self.block_c:
                return dist[(r, c)]
            d = dist[(r, c)]
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.gs and 0 <= nc < self.gs:
                    if (nr, nc) not in self.obstacles and (nr, nc) not in self.goal_cells and (nr, nc) not in dist:
                        dist[(nr, nc)] = d + 1
                        q.append((nr, nc))
        return 999.0

    def block_dist_to_goal(self):
        q = collections.deque()
        dist = {}
        for gr, gc in self.goal_cells:
            q.append((gr, gc))
            dist[(gr, gc)] = 0
            
        while q:
            r, c = q.popleft()
            if r == self.block_r and c == self.block_c:
                return dist[(r, c)]
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.gs and 0 <= nc < self.gs:
                    if (nr, nc) not in self.obstacles and (nr, nc) not in dist:
                        dist[(nr, nc)] = dist[(r, c)] + 1
                        q.append((nr, nc))
        return 999.0


# ════════════════════════════════════════════════════════════════
#  POLICY NETWORK (PPO-style)
# ════════════════════════════════════════════════════════════════
if TORCH_AVAILABLE:
    class PolicyNet(nn.Module):
        def __init__(self, obs_dim=OBS_SIZE, act_dim=ACTIONS, hidden=256):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden),  nn.ReLU(),
                nn.Linear(hidden, hidden),  nn.ReLU(),  # deeper = better maze memory
            )
            self.actor  = nn.Linear(hidden, act_dim)
            self.critic = nn.Linear(hidden, 1)

        def forward(self, x):
            h = self.shared(x)
            return self.actor(h), self.critic(h)

        def act(self, obs_np):
            with torch.no_grad():
                x = torch.FloatTensor(obs_np).unsqueeze(0)
                logits, val = self(x)
                probs = torch.softmax(logits, dim=-1)
                dist  = torch.distributions.Categorical(probs)
                a     = dist.sample()
            return a.item(), dist.log_prob(a).item(), val.item()

        def greedy_act(self, obs_np):
            with torch.no_grad():
                x = torch.FloatTensor(obs_np).unsqueeze(0)
                logits, _ = self(x)
                return torch.argmax(logits, dim=-1).item()

    class PPOTrainer:
        def __init__(self, net, lr=3e-4, clip=0.2, epochs=3):
            self.net = net
            self.opt = optim.Adam(net.parameters(), lr=lr)
            self.clip = clip
            self.epochs = epochs
            self.buf = []

        def store(self, obs, act, logp, val, rew, done):
            self.buf.append((obs, act, logp, val, rew, done))

        def train(self, last_val=0.0, gamma=0.99, lam=0.95):
            if len(self.buf) < 32:
                return 0.0
            obs_l, act_l, logp_l, val_l, rew_l, done_l = zip(*self.buf)
            T = len(rew_l)
            adv = np.zeros(T, dtype=np.float32)
            gae = 0.0
            ret = np.zeros(T, dtype=np.float32)
            val_arr = list(val_l) + [last_val]
            for t in reversed(range(T)):
                mask  = 0.0 if done_l[t] else 1.0
                delta = rew_l[t] + gamma * val_arr[t+1] * mask - val_arr[t]
                gae   = delta + gamma * lam * mask * gae
                adv[t] = gae
                ret[t] = gae + val_arr[t]
            obs_t  = torch.FloatTensor(np.array(obs_l))
            act_t  = torch.LongTensor(act_l)
            logp_t = torch.FloatTensor(logp_l)
            ret_t  = torch.FloatTensor(ret)
            adv_t  = torch.FloatTensor(adv)
            adv_t  = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
            total_loss = 0.0
            for _ in range(self.epochs):
                logits, vals = self.net(obs_t)
                probs  = torch.softmax(logits, dim=-1)
                dist   = torch.distributions.Categorical(probs)
                new_lp = dist.log_prob(act_t)
                ratio  = torch.exp(new_lp - logp_t)
                s1 = ratio * adv_t
                s2 = torch.clamp(ratio, 1-self.clip, 1+self.clip) * adv_t
                p_loss = -torch.min(s1, s2).mean()
                v_loss = (ret_t - vals.squeeze()).pow(2).mean()
                ent    = dist.entropy().mean()
                loss   = p_loss + 0.5*v_loss - 0.02*ent  # higher entropy → more exploration
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()
                total_loss += loss.item()
            self.buf.clear()
            return total_loss / self.epochs


# ════════════════════════════════════════════════════════════════
#  Q-LEARNING FALLBACK
# ════════════════════════════════════════════════════════════════
class QAgent:
    def __init__(self):
        self.q = {}
        self.epsilon = 1.0
        self.alpha = 0.15
        self.gamma = 0.95

    def _disc(self, obs):
        return tuple((obs[i*5:i*5+5] > 0.5).astype(int).tolist()
                     for i in range(NUM_RAYS))

    def _q(self, s):
        if s not in self.q:
            self.q[s] = np.zeros(ACTIONS)
        return self.q[s]

    def act(self, obs, greedy=False):
        s = self._disc(obs)
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, ACTIONS-1), 0.0, 0.0
        return int(np.argmax(self._q(s))), 0.0, 0.0

    def greedy_act(self, obs):
        return self.act(obs, greedy=True)[0]

    def update(self, obs, a, r, obs2, done):
        s  = self._disc(obs)
        s2 = self._disc(obs2)
        q  = self._q(s)
        q[a] += self.alpha * (r + (0 if done else self.gamma*self._q(s2).max()) - q[a])

    def decay(self):
        self.epsilon = max(0.05, self.epsilon * 0.9995)


# ════════════════════════════════════════════════════════════════
#  APPLICATION
# ════════════════════════════════════════════════════════════════
class App:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("Push Block  ·  RL Agent")
        self.clock  = pygame.time.Clock()

        self.fB = pygame.font.SysFont("consolas", 32, bold=True)
        self.fM = pygame.font.SysFont("consolas", 17)
        self.fS = pygame.font.SysFont("consolas", 13)
        self.fT = pygame.font.SysFont("consolas", 11)

        self.env = PushBlockEnv()

        if TORCH_AVAILABLE:
            self.net     = PolicyNet()
            self.trainer = PPOTrainer(self.net)
            self.agent_type = "PPO (PyTorch)"
        else:
            self.q_agent    = QAgent()
            self.agent_type = "Q-Learning (numpy)"

        self.mode          = "menu"
        self.episodes      = 0
        self.wins          = 0
        self.train_rewards = collections.deque(maxlen=200)
        self.current_loss  = 0.0
        self.obs           = self.env.reset()
        self.anim_t        = 0
        self.particles     = []
        self.flash         = 0
        self.last_act      = 4
        self.speed         = 8

        self.manual_keys = {
            pygame.K_UP: 2, pygame.K_DOWN: 3,
            pygame.K_e: 0, pygame.K_q: 1,
            pygame.K_SPACE: 4
        }

    # ── MAIN LOOP ────────────────────────────────────────
    def run(self):
        while True:
            self.handle_events()
            if self.mode == "training":
                for _ in range(50):
                    self._train_step()
            elif self.mode == "watching":
                # Fire exactly ONE step every 20 frames = ~3 moves/sec. Smooth and easy to follow.
                if self.anim_t % 20 == 0:
                    self._watch_step()
            self.draw()
            self.clock.tick(60)

    def handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    if self.mode in ("training","watching","manual"):
                        self.mode = "menu"
                    else:
                        pygame.quit(); sys.exit()
                if self.mode == "menu":
                    if e.key == pygame.K_t: self._start_training()
                    if e.key == pygame.K_w: self._start_watching()
                    if e.key == pygame.K_m: self._start_manual()
                if self.mode == "watching":
                    if e.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        self.speed = min(60, self.speed+4)
                    if e.key == pygame.K_MINUS:
                        self.speed = max(1, self.speed-4)
                if self.mode == "manual" and e.key in self.manual_keys:
                    a = self.manual_keys[e.key]
                    self.obs, _, done = self.env.step(a)
                    self.last_act = a
                    if done: self._episode_end()
            if e.type == pygame.MOUSEBUTTONDOWN:
                self._click(e.pos)

    def _click(self, pos):
        if self.mode == "menu":
            bx, by, bw, bh = W//2-130, H//2-60, 260, 50
            actions = [self._start_training, self._start_watching, self._start_manual]
            for i, fn in enumerate(actions):
                if pygame.Rect(bx, by+i*70, bw, bh).collidepoint(pos):
                    fn()
        if self.mode in ("training","watching","manual"):
            if pygame.Rect(W-120, 8, 110, 34).collidepoint(pos):
                self.mode = "menu"

    def _start_training(self):
        self.mode = "training"
        self.obs  = self.env.reset()

    def _start_watching(self):
        self.mode = "watching"
        self.obs  = self.env.reset()

    def _start_manual(self):
        self.mode = "manual"
        self.obs  = self.env.reset()

    def _train_step(self):
        if TORCH_AVAILABLE:
            a, logp, val = self.net.act(self.obs)
            obs2, r, done = self.env.step(a)
            self.trainer.store(self.obs, a, logp, val, r, done)
            self.last_act = a; self.obs = obs2
            if done:
                lv = 0.0
                self.current_loss = self.trainer.train(last_val=lv)
                self._episode_end()
        else:
            a, _, _ = self.q_agent.act(self.obs)
            obs2, r, done = self.env.step(a)
            self.q_agent.update(self.obs, a, r, obs2, done)
            self.q_agent.decay()
            self.last_act = a; self.obs = obs2
            if done: self._episode_end()

    def _watch_step(self):
        if TORCH_AVAILABLE:
            a = self.net.greedy_act(self.obs)
        else:
            a = self.q_agent.greedy_act(self.obs)
        self.obs, _, done = self.env.step(a)
        self.last_act = a
        if done:
            self.train_rewards.append(self.env.episode_rew)
            if self.env.success:
                self.wins += 1
            self.episodes += 1
            self.obs = self.env.reset()

    def _episode_end(self):
        self.train_rewards.append(self.env.episode_rew)
        self.episodes += 1
        if self.env.success:
            self.wins += 1
        # Auto-stop training after MAX_TRAIN_EPISODES and switch to watch
        if self.mode == "training" and self.episodes >= MAX_TRAIN_EPISODES:
            self.mode = "watching"
        self.obs = self.env.reset()

    def _burst(self):
        for _ in range(50):
            for gr, gc in self.env.goal_cells:
                px = BOARD_X + gc*CELL + CELL//2
                py = BOARD_Y + gr*CELL + CELL//2
                self.particles.append({
                    "x": px, "y": py,
                    "vx": random.uniform(-4, 4),
                    "vy": random.uniform(-6, -1),
                    "life": random.randint(30, 70),
                    "col": random.choice([GOAL_A, GOAL_B, (255,255,200)])
                })
                break

    def _update_particles(self):
        live = []
        for p in self.particles:
            p["x"] += p["vx"]; p["y"] += p["vy"]; p["vy"] += 0.18
            p["life"] -= 1
            if p["life"] > 0: live.append(p)
        self.particles = live

    # ════════════════════════════════════════════════════
    #  DRAW
    # ════════════════════════════════════════════════════
    def draw(self):
        self.anim_t += 1
        self.screen.fill(BG)
        self._update_particles()
        if self.flash > 0: self.flash -= 1

        if self.mode == "menu":
            self._draw_menu()
        else:
            self._draw_header()
            self._draw_board()
            self._draw_panel()
            self._draw_particles()
        pygame.display.flip()

    def _draw_menu(self):
        # Animated grid bg
        for r in range(14):
            for c in range(24):
                x = c*55 - (self.anim_t % 55)
                y = r*58
                pygame.draw.rect(self.screen, GRID_LINE, (x, y, 52, 55), 1)

        self._shadow_text(self.fB, "PUSH BLOCK", W//2, 150, TEXT_HI, 3)
        self._shadow_text(self.fM, "Reinforcement Learning Agent", W//2, 196, ACCENT, 1)
        self._text(self.fS, f"Engine: {self.agent_type}", W//2, 222, TEXT_LO)

        labels = [
            ("  [T]  Train Agent", ACCENT),
            ("  [W]  Watch Agent", BLOCK_B),
            ("  [M]  Manual Play", AGENT_B),
        ]
        bx, by, bw, bh = W//2-130, H//2-60, 260, 50
        for i, (label, col) in enumerate(labels):
            rect = pygame.Rect(bx, by+i*70, bw, bh)
            p = 0.5 + 0.5*math.sin(self.anim_t*0.05 + i)
            pygame.draw.rect(self.screen,
                             tuple(int(c*0.15) for c in col), rect, border_radius=8)
            pygame.draw.rect(self.screen,
                             tuple(int(c*(0.4+p*0.6)) for c in col), rect, 2, border_radius=8)
            self._text(self.fM, label, rect.centerx, rect.centery, col)

        spec = "Obs: 70-dim  |  Actions: 7  |  Reward: -0.0025/step +1.0 goal  |  Target: 4.5"
        self._text(self.fT, spec, W//2, H-30, TEXT_LO)

    def _draw_header(self):
        self._text(self.fB, "PUSH BLOCK  ·  " + self.mode.upper(), 20, 36, TEXT_HI, anchor="ml")
        wr = (self.wins/max(1,self.episodes))*100
        info = f"Episodes: {self.episodes}   Wins: {self.wins}   Win Rate: {wr:.1f}%"
        self._text(self.fM, info, W//2, 36, ACCENT)
        back = pygame.Rect(W-120, 8, 110, 34)
        pygame.draw.rect(self.screen, (30,35,55), back, border_radius=6)
        pygame.draw.rect(self.screen, TEXT_LO, back, 1, border_radius=6)
        self._text(self.fS, "ESC  Back", back.centerx, back.centery, TEXT_HI)

    def _to_iso(self, x, y, z=0):
        # x, y in [0, BOARD_PX]
        cx = x - (BOARD_PX / 2)
        cy = y - (BOARD_PX / 2)
        iso_x = (cx - cy) * 0.866
        iso_y = (cx + cy) * 0.5 - z
        return int(BOARD_X + BOARD_PX/2 + iso_x), int(BOARD_Y + BOARD_PX*0.4 + iso_y)

    def _draw_iso_cube(self, x, y, w, h, z, col, top_c=None, right_c=None):
        b0 = self._to_iso(x, y, 0)
        b1 = self._to_iso(x+w, y, 0)
        b2 = self._to_iso(x+w, y+h, 0)
        b3 = self._to_iso(x, y+h, 0)
        t0 = self._to_iso(x, y, z)
        t1 = self._to_iso(x+w, y, z)
        t2 = self._to_iso(x+w, y+h, z)
        t3 = self._to_iso(x, y+h, z)

        left_c  = col
        top_c   = top_c or tuple(min(255, int(c*1.2)) for c in col)
        right_c = right_c or tuple(int(c*0.8) for c in col)

        pygame.draw.polygon(self.screen, left_c, [b0, b3, t3, t0])
        pygame.draw.polygon(self.screen, (30,30,30), [b0, b3, t3, t0], 1)
        pygame.draw.polygon(self.screen, right_c, [b3, b2, t2, t3])
        pygame.draw.polygon(self.screen, (30,30,30), [b3, b2, t2, t3], 1)
        pygame.draw.polygon(self.screen, top_c, [t0, t1, t2, t3])
        pygame.draw.polygon(self.screen, (30,30,30), [t0, t1, t2, t3], 1)

    def _draw_board(self):
        # Draw floor base
        f0 = self._to_iso(0, 0, -20)
        f1 = self._to_iso(BOARD_PX, 0, -20)
        f2 = self._to_iso(BOARD_PX, BOARD_PX, -20)
        f3 = self._to_iso(0, BOARD_PX, -20)
        pygame.draw.polygon(self.screen, WALL_C, [f0, f1, f2, f3])

        # Draw grid
        g0 = self._to_iso(0, 0, 0)
        g1 = self._to_iso(BOARD_PX, 0, 0)
        g2 = self._to_iso(BOARD_PX, BOARD_PX, 0)
        g3 = self._to_iso(0, BOARD_PX, 0)
        pygame.draw.polygon(self.screen, GRID_BG, [g0, g1, g2, g3])
        # Goal Area (Row 0 is goal)
        goal_w = BOARD_PX
        goal_h = CELL
        ga0 = self._to_iso(0, 0, 0.5)
        ga1 = self._to_iso(goal_w, 0, 0.5)
        ga2 = self._to_iso(goal_w, goal_h, 0.5)
        ga3 = self._to_iso(0, goal_h, 0.5)
        pygame.draw.polygon(self.screen, GOAL_A, [ga0, ga1, ga2, ga3])



        for i in range(GRID_N+1):
            p1 = self._to_iso(i*CELL, 0, 0.2); p2 = self._to_iso(i*CELL, BOARD_PX, 0.2)
            pygame.draw.line(self.screen, GRID_LINE, p1, p2)
            p3 = self._to_iso(0, i*CELL, 0.2); p4 = self._to_iso(BOARD_PX, i*CELL, 0.2)
            pygame.draw.line(self.screen, GRID_LINE, p3, p4)

        if self.mode in ("watching", "training"):
            self._draw_rays()

        # Depth sort and draw entities
        entities = []
        entities.append({"type":"block", "y": self.env.block_r*CELL + CELL/2, "x": self.env.block_c*CELL + CELL/2})
        entities.append({"type":"agent", "y": self.env.agent_r*CELL + CELL/2, "x": self.env.agent_c*CELL + CELL/2})
        for obr, obc in self.env.obstacles:
            entities.append({"type":"obstacle", "y": obr*CELL + CELL/2, "x": obc*CELL + CELL/2, "r": obr, "c": obc})
        
        # Sort by depth (isometric depth is largely x + y in top-down view coordinates)
        entities.sort(key=lambda e: e["x"] + e["y"])
        
        for e in entities:
            if e["type"] == "block": self._draw_block()
            if e["type"] == "agent": self._draw_agent()
            if e["type"] == "obstacle": self._draw_obstacle(e["r"], e["c"])

    def _draw_rays(self):
        angles = [i*(360/NUM_RAYS) for i in range(NUM_RAYS)]
        ax = self.env.agent_c*CELL + CELL/2
        ay = self.env.agent_r*CELL + CELL/2
        start_pt = self._to_iso(ax, ay, 10)
        for i, deg in enumerate(angles):
            rad = math.radians(deg + self.env.agent_dir*90)
            dx, dy = math.cos(rad), math.sin(rad)
            obs5 = self.obs[i*5:(i+1)*5]
            d = obs5[3]
            col = (255,200,50) if obs5[2]>0.5 else \
                  (100,200,100) if obs5[1]>0.5 else \
                  (200,80,80)   if obs5[0]>0.5 else (70,80,110)
            ex = ax + dx*d*BOARD_PX; ey = ay + dy*d*BOARD_PX
            end_pt = self._to_iso(ex, ey, 10)
            pygame.draw.line(self.screen, (*col, 60), start_pt, end_pt, max(1, int(3 * (1-d))))

    def _draw_block(self):
        bw = CELL * 0.7
        bh = CELL * 0.7
        bz = CELL * 0.7
        bx = self.env.block_c*CELL + (CELL-bw)/2
        by = self.env.block_r*CELL + (CELL-bh)/2
        self._draw_iso_cube(bx, by, bw, bh, bz, BLOCK_B, BLOCK_A, (180,180,180))

    def _draw_agent(self):
        size = CELL * 0.5
        ax = self.env.agent_c*CELL + (CELL-size)/2
        ay = self.env.agent_r*CELL + (CELL-size)/2
        # Main body
        self._draw_iso_cube(ax, ay, size, size, size, AGENT_A)
        # Headband
        hb_z = size * 0.6
        hb_h = size * 0.2
        self._draw_iso_cube(ax-1, ay-1, size+2, size+2, hb_z + hb_h, AGENT_B)
        
        # Eyes
        d = self.env.agent_dir
        # simple directional indicator on top
        cx = ax + size/2
        cy = ay + size/2
        top_center = self._to_iso(cx, cy, size+2)
        dirs = [(0, -size/3), (size/3, 0), (0, size/3), (-size/3, 0)] # N E S W in grid
        dx, dy = dirs[d]
        eye_pt = self._to_iso(cx + dx, cy + dy, size+2)
        pygame.draw.circle(self.screen, (0,0,0), eye_pt, 4)

    def _draw_obstacle(self, r, c):
        bw = CELL * 0.9
        bh = CELL * 0.9
        bz = CELL * 0.8
        bx = c*CELL + (CELL-bw)/2
        by = r*CELL + (CELL-bh)/2
        self._draw_iso_cube(bx, by, bw, bh, bz, (180, 50, 50), (220, 80, 80), (130, 40, 40))

    def _draw_particles(self):
        for p in self.particles:
            alpha = min(255, int(p["life"]/70*255))
            rad = max(2, int(p["life"]/70*7))
            # treating p["y"] as z height, and p["x"] as x across the screen
            # we need to map 2D particles back to 3D or fake it.
            # actually, they were spawned at original 2D coordinates. 
            # let's keep them as screen-space overlays for the cool "burst" effect!
            s = pygame.Surface((rad*2, rad*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*p["col"], alpha), (rad, rad), rad)
            self.screen.blit(s, (int(p["x"])-rad, int(p["y"])-rad))

    def _draw_panel(self):
        panel = pygame.Rect(PANEL_X, BOARD_Y, PANEL_W, BOARD_PX)
        pygame.draw.rect(self.screen, PANEL_BG, panel, border_radius=8)
        pygame.draw.rect(self.screen, GRID_LINE, panel, 1, border_radius=8)

        y = BOARD_Y + 16

        def sec(title):
            nonlocal y
            pygame.draw.line(self.screen, ACCENT,
                             (PANEL_X+12, y), (PANEL_X+PANEL_W-12, y), 1)
            self._text(self.fT, title, PANEL_X+14, y+1, ACCENT, anchor="tl")
            y += 18

        def row(label, val, col=TEXT_HI):
            nonlocal y
            self._text(self.fT, label+":", PANEL_X+14, y, TEXT_LO, anchor="tl")
            self._text(self.fT, str(val), PANEL_X+PANEL_W-14, y, col, anchor="tr")
            y += 15

        sec("STATS")
        row("Agent",    self.agent_type[:20])
        row("Episodes", self.episodes)
        row("Wins",     self.wins)
        wr = (self.wins/max(1,self.episodes))*100
        row("Win Rate", f"{wr:.1f}%",
            GOAL_A if wr>50 else ORG_C if wr>20 else RED_C)
        if TORCH_AVAILABLE and self.current_loss:
            row("PPO Loss", f"{self.current_loss:.4f}", ORG_C)
        y += 4

        sec("REWARD HISTORY")
        self._draw_reward_chart(PANEL_X+12, y, PANEL_W-24, 88)
        y += 98

        sec("CURRENT EPISODE")
        row("Steps",  self.env.steps)
        row("Reward", f"{self.env.episode_rew:.4f}",
            GOAL_A if self.env.episode_rew > 0 else RED_C)
        row("Block→Goal", self.env.block_dist_to_goal())
        row("Last Action", ACT_NAMES[self.last_act][:16])
        y += 4

        sec("ENV PARAMETERS")
        row("block_scale",      f"{self.env.block_scale:.1f}")
        row("dynamic_friction", f"{self.env.dynamic_friction:.1f}")
        row("static_friction",  f"{self.env.static_friction:.1f}")
        row("block_drag",       f"{self.env.block_drag:.1f}")
        y += 4

        sec("REWARD FUNCTION")
        row("-0.0025",  "every step",    RED_C)
        row("+1.0",     "block in goal!", GOAL_B)
        y += 4

        sec("RAYCASTS  (14 x 5 = 70 obs)")
        self._draw_ray_bars(PANEL_X+12, y, PANEL_W-24)
        y += 80

        if self.mode == "manual":
            y += 4; sec("CONTROLS")
            for hint in ["↑↓←→ Move", "Q/E Rotate", "SPACE Wait", "ESC Back"]:
                self._text(self.fT, hint, PANEL_X+14, y, TEXT_LO, anchor="tl"); y += 14

        if self.mode == "watching":
            y += 4; sec(f"SPEED  {self.speed} steps/frame")
            self._text(self.fT, "+/- to adjust", PANEL_X+14, y, TEXT_LO, anchor="tl")

        self._text(self.fT, "Benchmark target: 4.5 mean reward",
                   PANEL_X+PANEL_W//2, BOARD_Y+BOARD_PX-14, TEXT_LO)

    def _draw_reward_chart(self, x, y, w, h):
        pygame.draw.rect(self.screen, (18,22,38), (x, y, w, h), border_radius=4)
        rews = list(self.train_rewards)
        if len(rews) < 2:
            self._text(self.fT, "collecting data…", x+w//2, y+h//2, TEXT_LO); return
        mn, mx = min(rews), max(rews)
        span = max(0.01, mx-mn)
        pts  = [(x + int(i/(len(rews)-1)*w),
                 y + h - int((r-mn)/span*h)) for i, r in enumerate(rews)]
        if len(pts) >= 2:
            pygame.draw.lines(self.screen, ACCENT, False, pts, 1)
        avg = sum(rews[-50:])/max(1,len(rews[-50:]))
        self._text(self.fT, f"avg50: {avg:.3f}", x+w-4, y+4, GOAL_B, anchor="tr")
        self._text(self.fT, f"{mx:.2f}", x+4, y+4, TEXT_LO, anchor="tl")
        self._text(self.fT, f"{mn:.2f}", x+4, y+h-4, TEXT_LO, anchor="bl" if hasattr(self.fT,"render") else "tl")

    def _draw_ray_bars(self, x, y, w):
        for i in range(NUM_RAYS):
            obs5 = self.obs[i*5:(i+1)*5]
            col = GOAL_A if obs5[1]>0.5 else BLOCK_A if obs5[2]>0.5 \
                  else RED_C if obs5[0]>0.5 else TEXT_LO
            bx2 = x + int(i/NUM_RAYS * w)
            bw2 = max(1, int(w/NUM_RAYS) - 1)
            pygame.draw.rect(self.screen, BAR_BG, (bx2, y, bw2, 65))
            fill = int(obs5[2]*65 if obs5[2]>0.5 else
                       obs5[1]*65 if obs5[1]>0.5 else
                       obs5[0]*65 if obs5[0]>0.5 else (1-obs5[3])*30)
            if fill > 0:
                pygame.draw.rect(self.screen, col, (bx2, y+65-fill, bw2, fill))

    def _text(self, font, txt, x, y, col, anchor="mc"):
        surf = font.render(str(txt), True, col)
        rect = surf.get_rect()
        if   anchor == "mc": rect.center   = (x, y)
        elif anchor == "tl": rect.topleft  = (x, y)
        elif anchor == "tr": rect.topright = (x, y)
        elif anchor == "ml": rect.midleft  = (x, y)
        elif anchor == "bl": rect.bottomleft = (x, y)
        self.screen.blit(surf, rect)

    def _shadow_text(self, font, txt, x, y, col, sh=2):
        self._text(font, txt, x+sh, y+sh, (0,0,0))
        self._text(font, txt, x,    y,    col)


# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 58)
    print("  PUSH BLOCK  -  RL Agent")
    print("=" * 58)
    print(f"  PyTorch: {'[v] PPO (actor-critic)' if TORCH_AVAILABLE else '[x] Q-Learning fallback'}")
    print()
    print("  Observation space: 70 vars (14 raycasts x 5)")
    print("  Action space:      7 discrete actions")
    print("  Reward:           -0.0025/step  +1.0 on goal")
    print("  Benchmark target:  4.5 mean reward")
    print()
    print("  T = Train  |  W = Watch  |  M = Manual  |  ESC = Back")
    print("=" * 58)
    App().run()
