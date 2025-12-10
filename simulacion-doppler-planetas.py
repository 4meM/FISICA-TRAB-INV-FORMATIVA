import pygame
import numpy as np
import math
import random
from collections import deque

WIDTH, HEIGHT = 1400, 900
BG_COLOR = (8, 8, 12)

# Physics Constants
G = 1.8
SOLAR_MASS = 5000
DT = 0.025

# Doppler Constants
C_SIGNAL = 300.0
FREQ_BASE = 100.0
COLLISION_RADIUS = 15.0
DOPPLER_HISTORY_SIZE = 150
PREDICTION_STEPS = 300

# Perturbation Constants
PERTURBATION_INTENSITY = 2.5
PERTURBATION_ASTEROID_MASS = 3000

class DopplerAnalyzer:
    """Analyzes Doppler effect from Earth to predict collisions."""
    def __init__(self, earth, body1, body2):
        self.earth = earth
        self.body1 = body1
        self.body2 = body2
        self.history = deque(maxlen=DOPPLER_HISTORY_SIZE)
        self.collision_predicted = False
        self.collision_imminent = False
        self.time_to_collision = float('inf')
        self.predicted_collision_point = None
        self.collision_frame = None
        
        self.baseline_doppler_b1 = None
        self.baseline_doppler_b2 = None
        self.orbital_anomaly_detected = False
        self.anomaly_level = 0  # 0: normal, 1: mild, 2: moderate, 3: severe
        
    def calculate_doppler_shift(self):
        """Calculates Doppler shift observed from Earth."""
        diff = self.body2.pos - self.body1.pos
        dist = np.linalg.norm(diff)
        
        if dist < 0.1:
            return 0, 0, 0, 0, dist, True
        
        # Doppler from Earth to Body 1
        diff_earth_b1 = self.body1.pos - self.earth.pos
        dist_earth_b1 = np.linalg.norm(diff_earth_b1)
        
        if dist_earth_b1 > 0.1:
            unit_earth_b1 = diff_earth_b1 / dist_earth_b1
            vel_radial_b1 = np.dot(self.body1.vel - self.earth.vel, unit_earth_b1)
            # Doppler Formula: f_obs = f_0 * (c + v_obs) / c
            freq_b1 = FREQ_BASE * (C_SIGNAL + vel_radial_b1) / C_SIGNAL
        else:
            vel_radial_b1 = 0
            freq_b1 = FREQ_BASE
        
        # Doppler from Earth to Body 2
        diff_earth_b2 = self.body2.pos - self.earth.pos
        dist_earth_b2 = np.linalg.norm(diff_earth_b2)
        
        if dist_earth_b2 > 0.1:
            unit_earth_b2 = diff_earth_b2 / dist_earth_b2
            vel_radial_b2 = np.dot(self.body2.vel - self.earth.vel, unit_earth_b2)
            freq_b2 = FREQ_BASE * (C_SIGNAL + vel_radial_b2) / C_SIGNAL
        else:
            vel_radial_b2 = 0
            freq_b2 = FREQ_BASE
        
        vel_rel = self.body2.vel - self.body1.vel
        unit_vec = diff / dist
        radial_velocity = np.dot(vel_rel, unit_vec)
        
        doppler_shift_b1 = freq_b1 - FREQ_BASE
        doppler_shift_b2 = freq_b2 - FREQ_BASE
        
        self.history.append({
            'dist': dist,
            'radial_vel': radial_velocity,
            'doppler_b1': doppler_shift_b1,
            'doppler_b2': doppler_shift_b2,
            'vel_radial_b1': vel_radial_b1,
            'vel_radial_b2': vel_radial_b2
        })
        
        if self.baseline_doppler_b1 is None and len(self.history) > 20:
            self.baseline_doppler_b1 = np.mean([h['doppler_b1'] for h in self.history])
            self.baseline_doppler_b2 = np.mean([h['doppler_b2'] for h in self.history])
        
        self.detect_orbital_anomaly(doppler_shift_b1, doppler_shift_b2, vel_radial_b1, vel_radial_b2)
        self.analyze_collision_risk(dist, radial_velocity)
        
        return doppler_shift_b1, doppler_shift_b2, vel_radial_b1, vel_radial_b2, dist, self.collision_imminent
    
    def detect_orbital_anomaly(self, doppler_b1, doppler_b2, vrad_b1, vrad_b2):
        if self.baseline_doppler_b1 is None:
            self.orbital_anomaly_detected = False
            self.anomaly_level = 0
            return
        
        deviation_b1 = abs(doppler_b1 - self.baseline_doppler_b1)
        deviation_b2 = abs(doppler_b2 - self.baseline_doppler_b2)
        
        abnormal_vel_b1 = abs(vrad_b1) > 3.0
        abnormal_vel_b2 = abs(vrad_b2) > 3.0
        
        if len(self.history) > 30:
            recent_doppler_b1 = [h['doppler_b1'] for h in list(self.history)[-30:]]
            recent_doppler_b2 = [h['doppler_b2'] for h in list(self.history)[-30:]]
            
            doppler_change_b1 = max(recent_doppler_b1) - min(recent_doppler_b1)
            doppler_change_b2 = max(recent_doppler_b2) - min(recent_doppler_b2)
            
            if (deviation_b1 > 15 or deviation_b2 > 15) or (abnormal_vel_b1 or abnormal_vel_b2):
                self.orbital_anomaly_detected = True
                self.anomaly_level = 3
            elif (deviation_b1 > 8 or deviation_b2 > 8) or (doppler_change_b1 > 10 or doppler_change_b2 > 10):
                self.orbital_anomaly_detected = True
                self.anomaly_level = 2
            elif deviation_b1 > 4 or deviation_b2 > 4:
                self.orbital_anomaly_detected = True
                self.anomaly_level = 1
            else:
                self.orbital_anomaly_detected = False
                self.anomaly_level = 0
        else:
            self.orbital_anomaly_detected = False
            self.anomaly_level = 0
    
    def analyze_collision_risk(self, current_dist, radial_velocity):
        collision_distance = self.body1.radius + self.body2.radius + 5
        
        if current_dist < collision_distance * 2 and radial_velocity < -0.5:
            self.collision_imminent = True
            if radial_velocity < 0:
                self.time_to_collision = abs(current_dist / radial_velocity)
            else:
                self.time_to_collision = 0
        elif current_dist < collision_distance * 5 and radial_velocity < -1.0:
            self.collision_imminent = False
            if radial_velocity < 0:
                self.time_to_collision = abs(current_dist / radial_velocity)
            else:
                self.time_to_collision = float('inf')
        else:
            self.collision_imminent = False
            if radial_velocity < 0:
                self.time_to_collision = abs(current_dist / radial_velocity)
            else:
                self.time_to_collision = float('inf')
    
    def predict_collision_using_doppler(self, all_bodies):
        if len(self.history) < 10:
            return False, None, None
        
        recent_vels = [h['radial_vel'] for h in list(self.history)[-10:]]
        approaching = sum(1 for v in recent_vels if v < -0.2) > 6
        
        if not approaching:
            return False, None, None
        
        sim_pos1 = self.body1.pos.copy()
        sim_vel1 = self.body1.vel.copy()
        sim_pos2 = self.body2.pos.copy()
        sim_vel2 = self.body2.vel.copy()
        
        for step in range(PREDICTION_STEPS):
            acc1 = np.zeros(2, dtype=float)
            acc2 = np.zeros(2, dtype=float)
            
            for other in all_bodies:
                if other == self.body1 or other == self.body2:
                    continue
                
                diff1 = other.pos - sim_pos1
                dist1 = np.linalg.norm(diff1)
                if dist1 > 1:
                    f_mag1 = G * self.body1.mass * other.mass / (dist1**2)
                    acc1 += (diff1 / dist1) * f_mag1 / self.body1.mass
                
                diff2 = other.pos - sim_pos2
                dist2 = np.linalg.norm(diff2)
                if dist2 > 1:
                    f_mag2 = G * self.body2.mass * other.mass / (dist2**2)
                    acc2 += (diff2 / dist2) * f_mag2 / self.body2.mass
            
            diff_mutual = sim_pos2 - sim_pos1
            dist_mutual = np.linalg.norm(diff_mutual)
            
            if dist_mutual > 1:
                f_mag = G * self.body1.mass * self.body2.mass / (dist_mutual**2)
                acc1 += (diff_mutual / dist_mutual) * f_mag / self.body1.mass
                acc2 -= (diff_mutual / dist_mutual) * f_mag / self.body2.mass
            
            sim_vel1 += acc1 * DT * 2
            sim_vel2 += acc2 * DT * 2
            sim_pos1 += sim_vel1 * DT * 2
            sim_pos2 += sim_vel2 * DT * 2
            
            collision_dist = np.linalg.norm(sim_pos2 - sim_pos1)
            min_collision_dist = self.body1.radius + self.body2.radius + 2
            if collision_dist < min_collision_dist:
                collision_point = (sim_pos1 + sim_pos2) / 2
                return True, collision_point, step
        
        return False, None, None
    
    def get_warning_level(self):
        earth_involved = (self.body1.name == "Tierra" or self.body2.name == "Tierra")
        
        if self.collision_imminent:
            return 3
        elif self.collision_predicted:
            return 3 if earth_involved else 2
        elif self.time_to_collision < 50 or self.orbital_anomaly_detected:
            return 2 if earth_involved else 1
        else:
            return 0
    
    def involves_earth(self):
        return self.body1.name == "Tierra" or self.body2.name == "Tierra"

class Body:
    def __init__(self, name, mass, dist_from_sun, color, radius, is_sun=False, is_station=False):
        self.name = name
        self.mass = mass
        self.radius = radius
        self.color = color
        self.is_sun = is_sun
        self.is_station = is_station
        self.dist_from_sun = dist_from_sun

        if is_sun:
            self.pos = np.array([0, 0], dtype=float)
            self.vel = np.array([0, 0], dtype=float)
            self.orbital_angle = 0
            self.angular_velocity = 0
        else:
            self.orbital_angle = random.uniform(0, 2 * math.pi)
            # Angular velocity: omega = sqrt(G*M/r^3)
            self.angular_velocity = math.sqrt(G * SOLAR_MASS / (dist_from_sun ** 3))
            
            px = math.cos(self.orbital_angle) * dist_from_sun
            py = math.sin(self.orbital_angle) * dist_from_sun
            self.pos = np.array([px, py], dtype=float)

            v_orbital = math.sqrt(G * SOLAR_MASS / dist_from_sun)
            vx = -math.sin(self.orbital_angle) * v_orbital
            vy = math.cos(self.orbital_angle) * v_orbital
            self.vel = np.array([vx, vy], dtype=float)

        self.acc = np.zeros(2, dtype=float)
        self.trail = []
        self.perturbed = False
        self.use_circular_orbit = True
        self.future_positions = []
        
    def draw_orbit_guide(self, surface, scale, offset):
        if self.is_sun: return
        
        radius_screen = int(self.dist_from_sun * scale)
        if radius_screen > 0:
            pygame.draw.circle(surface, (30, 30, 40), (int(offset[0]), int(offset[1])), radius_screen, 1)

    def draw(self, surface, scale, offset, is_selected=False):
        x = int(self.pos[0] * scale + offset[0])
        y = int(self.pos[1] * scale + offset[1])
        
        if is_selected:
            pulse = abs(math.sin(pygame.time.get_ticks() / 300))
            ring_radius = int(self.radius * 2.5 + pulse * 8)
            ring_color = (0, 255, 200)
            pygame.draw.circle(surface, ring_color, (x, y), ring_radius, 3)
            pygame.draw.circle(surface, ring_color, (x, y), self.radius + 3, 2)
        
        if len(self.future_positions) > 1:
            points = []
            for i, (px, py) in enumerate(self.future_positions):
                sx = int(px * scale + offset[0])
                sy = int(py * scale + offset[1])
                points.append((sx, sy))
            if len(points) > 1:
                for i in range(0, len(points) - 1, 3):
                    if i + 1 < len(points):
                        pygame.draw.line(surface, (255, 150, 0), points[i], points[i+1], 2)
        
        if len(self.trail) > 2:
            points = []
            for tx, ty in self.trail:
                sx = int(tx * scale + offset[0])
                sy = int(ty * scale + offset[1])
                points.append((sx, sy))
            if len(points) > 1:
                pygame.draw.lines(surface, self.color, False, points, 1)

        r_visual = max(3, int(self.radius * (scale/15)))
        
        if self.is_station:
            pygame.draw.circle(surface, self.color, (x, y), r_visual)
            pygame.draw.circle(surface, (255, 255, 255), (x, y), r_visual + 3, 1)
            pygame.draw.line(surface, (100, 255, 100), (x, y), (x, y - r_visual * 3), 2)
        else:
            if self.perturbed:
                pygame.draw.circle(surface, (255, 50, 50), (x, y), r_visual + 5, 2)
            pygame.draw.circle(surface, self.color, (x, y), r_visual)

    def predict_trajectory(self, bodies, steps=50):
        if self.is_sun:
            return []
        
        test_pos = self.pos.copy()
        test_vel = self.vel.copy()
        predictions = []
        
        for _ in range(steps):
            test_acc = np.zeros(2, dtype=float)
            for other in bodies:
                if other == self:
                    continue
                diff = other.pos - test_pos
                dist = np.linalg.norm(diff)
                if dist < 1:
                    continue
                f_mag = G * self.mass * other.mass / (dist**2)
                f_vec = (diff / dist) * f_mag
                test_acc += f_vec / self.mass
            
            test_vel += test_acc * DT * 3
            test_pos += test_vel * DT * 3
            predictions.append(test_pos.copy())
        
        return predictions
    
    def update_physics(self, bodies):
        if self.is_sun: return

        if self.use_circular_orbit and not self.perturbed:
            self.orbital_angle += self.angular_velocity * DT
            
            self.pos[0] = math.cos(self.orbital_angle) * self.dist_from_sun
            self.pos[1] = math.sin(self.orbital_angle) * self.dist_from_sun
            
            v_orbital = self.angular_velocity * self.dist_from_sun
            self.vel[0] = -math.sin(self.orbital_angle) * v_orbital
            self.vel[1] = math.cos(self.orbital_angle) * v_orbital
            
        else:
            self.acc = np.zeros(2, dtype=float)
            
            for other in bodies:
                if other == self: continue
                
                diff = other.pos - self.pos
                dist = np.linalg.norm(diff)
                
                if dist < 1: continue
                
                # Gravity Force: F = G * m1 * m2 / r^2
                f_mag = G * self.mass * other.mass / (dist**2)
                f_vec = (diff / dist) * f_mag
                
                self.acc += f_vec / self.mass

            self.vel += self.acc * DT
            self.pos += self.vel * DT
        
        if random.random() < 0.2: 
            self.trail.append(self.pos.copy())
            if len(self.trail) > 100: self.trail.pop(0)

def configure_collision_trajectory(target_body, destination_body, severity):
    """Simulates Gravity Assist perturbation."""
    direction = destination_body.pos - target_body.pos
    dist = np.linalg.norm(direction)
    
    if dist > 1:
        direction_norm = direction / dist
        
        # Escape velocity: v_escape = sqrt(2GM/r)
        v_escape_destination = math.sqrt(2 * G * destination_body.mass / dist)
        velocity_change = v_escape_destination * severity * 2.5
        
        target_body.vel += direction_norm * velocity_change
        target_body.perturbed = True
        
        return True
    return False

def apply_gravitational_perturbation(target_body, bodies):
    """Simulates close encounter with massive object."""
    angle = random.uniform(0, 2 * math.pi)
    encounter_distance = random.uniform(30, 80)
    
    asteroid_pos = target_body.pos + np.array([
        math.cos(angle) * encounter_distance,
        math.sin(angle) * encounter_distance
    ])
    
    diff = asteroid_pos - target_body.pos
    dist = np.linalg.norm(diff)
    
    if dist > 1:
        f_mag = G * target_body.mass * PERTURBATION_ASTEROID_MASS / (dist**2)
        f_vec = (diff / dist) * f_mag
        
        impulse_duration = PERTURBATION_INTENSITY
        impulse = (f_vec / target_body.mass) * impulse_duration
        
        target_body.vel += impulse
        target_body.perturbed = True
        
        return asteroid_pos
    return None

def draw_doppler_panel(screen, analyzers, font, font_small, perturbed_body, earth):
    panel_x = WIDTH - 420
    panel_y = 10
    panel_width = 410
    panel_height = min(550, HEIGHT - 20)
    
    panel_surface = pygame.Surface((panel_width, panel_height))
    panel_surface.set_alpha(220)
    panel_surface.fill((15, 15, 25))
    screen.blit(panel_surface, (panel_x, panel_y))
    
    pygame.draw.rect(screen, (100, 150, 255), (panel_x, panel_y, panel_width, panel_height), 2)
    
    title = font.render("ANÁLISIS DOPPLER", True, (100, 200, 255))
    screen.blit(title, (panel_x + 10, panel_y + 10))
    
    y_offset = panel_y + 45
    
    if perturbed_body:
        perturb_text = font_small.render(f"Planeta Perturbado: {perturbed_body.name}", True, (255, 200, 100))
        screen.blit(perturb_text, (panel_x + 15, y_offset))
        y_offset += 30
    
    pygame.draw.line(screen, (100, 150, 255), (panel_x + 10, y_offset), (panel_x + panel_width - 10, y_offset), 1)
    y_offset += 15
    
    collision_detected = False
    shown_count = 0
    
    # Earth Alerts
    earth_alerts = []
    for analyzer in analyzers:
        if not analyzer.involves_earth():
            continue
        doppler_b1, doppler_b2, vrad_b1, vrad_b2, dist, warning = analyzer.calculate_doppler_shift()
        warning_level = analyzer.get_warning_level()
        
        if warning_level >= 2 or analyzer.orbital_anomaly_detected:
            other = analyzer.body2 if analyzer.body1.name == "Tierra" else analyzer.body1
            earth_alerts.append((analyzer, other, doppler_b1, doppler_b2, vrad_b1, vrad_b2, dist, warning_level))
    
    if earth_alerts:
        pulse = abs(math.sin(pygame.time.get_ticks() / 200))
        alert_bg = pygame.Surface((panel_width - 20, 80))
        alert_bg.set_alpha(int(150 + pulse * 100))
        alert_bg.fill((100, 0, 0))
        screen.blit(alert_bg, (panel_x + 10, y_offset))
        
        earth_title = font.render("🌍 ¡TIERRA EN PELIGRO!", True, (255, 255, 0))
        screen.blit(earth_title, (panel_x + 20, y_offset + 5))
        y_offset += 30
        
        for analyzer, other, doppler_b1, doppler_b2, vrad_b1, vrad_b2, dist, warning_level in earth_alerts[:1]:
            if warning_level == 3:
                msg = f"COLISIÓN con {other.name} - EVACUACIÓN"
            elif analyzer.collision_predicted:
                msg = f"Colisión predicha con {other.name} en {analyzer.time_to_collision:.0f}s"
            else:
                msg = f"{other.name} se salió de órbita - Acercamiento detectado"
            
            alert_text = font_small.render(msg, True, (255, 255, 255))
            screen.blit(alert_text, (panel_x + 20, y_offset))
            y_offset += 20
            
            doppler_earth = doppler_b1 if analyzer.body1.name == "Tierra" else doppler_b2
            info = font_small.render(f"Doppler Tierra: {doppler_earth:+.1f}Hz | Dist: {dist:.1f}u", True, (255, 200, 200))
            screen.blit(info, (panel_x + 20, y_offset))
            y_offset += 25
            shown_count += 1
        
        y_offset += 10
    
    # Orbital Anomalies
    anomalies = []
    for analyzer in analyzers:
        if analyzer.body1.is_sun or analyzer.body2.is_sun:
            continue
        if analyzer.involves_earth():
            continue
        doppler_b1, doppler_b2, vrad_b1, vrad_b2, dist, warning = analyzer.calculate_doppler_shift()
        
        if analyzer.orbital_anomaly_detected and analyzer.anomaly_level >= 2:
            anomalies.append((analyzer, doppler_b1, doppler_b2, vrad_b1, vrad_b2))
    
    if anomalies:
        anomaly_title = font_small.render("🛸 ANOMALÍAS ORBITALES:", True, (255, 200, 0))
        screen.blit(anomaly_title, (panel_x + 15, y_offset))
        y_offset += 25
        
        for analyzer, doppler_b1, doppler_b2, vrad_b1, vrad_b2 in anomalies[:2]:
            if analyzer.anomaly_level == 3:
                status = "SEVERA - Planeta fuera de órbita"
                color = (255, 100, 0)
            else:
                status = "MODERADA - Desviación detectada"
                color = (255, 180, 0)
            
            if analyzer.body1.perturbed:
                anomaly_body = analyzer.body1
                doppler_val = doppler_b1
                vrad_val = vrad_b1
            elif analyzer.body2.perturbed:
                anomaly_body = analyzer.body2
                doppler_val = doppler_b2
                vrad_val = vrad_b2
            else:
                continue
            
            text_lines = [
                f"• {anomaly_body.name}: {status}",
                f"  Doppler: {doppler_val:+.1f}Hz (anormal)",
                f"  Vel.radial: {vrad_val:+.2f}u/s"
            ]
            
            for line in text_lines:
                text = font_small.render(line, True, color)
                screen.blit(text, (panel_x + 20, y_offset))
                y_offset += 18
            
            y_offset += 8
            shown_count += 1
        
        y_offset += 10
    
    # Critical Alerts
    critical_pairs = []
    for analyzer in analyzers:
        if analyzer.body1.is_sun or analyzer.body2.is_sun:
            continue
        
        doppler_b1, doppler_b2, vrad_b1, vrad_b2, dist, warning = analyzer.calculate_doppler_shift()
        warning_level = analyzer.get_warning_level()
        
        if warning_level >= 2:
            critical_pairs.append((analyzer, doppler_b1, doppler_b2, vrad_b1, vrad_b2, dist, warning, warning_level))
            collision_detected = True
    
    if critical_pairs:
        alert_title = font_small.render("⚠ ALERTAS DE COLISIÓN:", True, (255, 100, 100))
        screen.blit(alert_title, (panel_x + 15, y_offset))
        y_offset += 25
        
        for analyzer, doppler_b1, doppler_b2, vrad_b1, vrad_b2, dist, warning, warning_level in critical_pairs:
            pair_name = f"{analyzer.body1.name} ↔ {analyzer.body2.name}"
            
            min_collision_dist = analyzer.body1.radius + analyzer.body2.radius + 5
            
            if warning_level == 3:
                color = (255, 50, 50)
                status = f"¡COLISIÓN FÍSICA! Dist: {dist:.1f}u (mín: {min_collision_dist:.0f}u)"
            elif warning_level == 2:
                color = (255, 150, 0)
                status = f"Colisión predicha en ~{analyzer.time_to_collision:.1f}s"
            else:
                color = (255, 200, 0)
                status = f"Acercamiento peligroso - Dist: {dist:.1f}u"
            
            name_surf = font_small.render(pair_name, True, color)
            screen.blit(name_surf, (panel_x + 20, y_offset))
            y_offset += 20
            
            info_lines = [
                f"  {status}",
                f"  Doppler {analyzer.body1.name}: {doppler_b1:+.1f}Hz",
                f"  Doppler {analyzer.body2.name}: {doppler_b2:+.1f}Hz"
            ]
            
            if analyzer.collision_frame:
                info_lines.append(f"  Colisión en: ~{analyzer.collision_frame * DT * 2:.1f}s")
                color = (255, 50, 50)
                status = "¡COLISIÓN!"
            else:
                color = (255, 150, 0)
                status = "RIESGO ALTO"
            
            name_surf = font_small.render(pair_name, True, color)
            screen.blit(name_surf, (panel_x + 20, y_offset))
            y_offset += 20
            
            info_lines = [
                f"  {status} - Dist: {dist:.1f}u",
                f"  Doppler {analyzer.body1.name}: {doppler_b1:+.1f}Hz",
                f"  Doppler {analyzer.body2.name}: {doppler_b2:+.1f}Hz",
                f"  T. Colisión: {analyzer.time_to_collision:.1f}s"
            ]
            
            for line in info_lines:
                text = font_small.render(line, True, (220, 220, 220))
                screen.blit(text, (panel_x + 20, y_offset))
                y_offset += 18
            
            y_offset += 10
            shown_count += 1
    
    # Monitoring perturbed planet
    if perturbed_body and shown_count < 3:
        y_offset += 5
        monitor_title = font_small.render(f"Monitoreo {perturbed_body.name}:", True, (150, 200, 255))
        screen.blit(monitor_title, (panel_x + 15, y_offset))
        y_offset += 25
        
        for analyzer in analyzers:
            if shown_count >= 5:
                break
                
            if analyzer.body1.is_sun or analyzer.body2.is_sun:
                continue
            
            if analyzer.body1 != perturbed_body and analyzer.body2 != perturbed_body:
                continue
            
            doppler_b1, doppler_b2, vrad_b1, vrad_b2, dist, warning = analyzer.calculate_doppler_shift()
            warning_level = analyzer.get_warning_level()
            
            if warning_level >= 2:
                continue
            
            other_body = analyzer.body2 if analyzer.body1 == perturbed_body else analyzer.body1
            is_perturbed_b1 = (analyzer.body1 == perturbed_body)
            
            doppler_target = doppler_b1 if is_perturbed_b1 else doppler_b2
            vrad_target = vrad_b1 if is_perturbed_b1 else vrad_b2
            
            if abs(vrad_target) > 0.5:
                color = (255, 200, 100) if vrad_target > 0 else (100, 200, 255)
                status = "alejándose" if vrad_target > 0 else "acercándose"
            else:
                color = (150, 150, 150)
                status = "estable"
            
            line_text = f"• {other_body.name}: Δf={doppler_target:+.1f}Hz ({status})"
            text = font_small.render(line_text, True, color)
            screen.blit(text, (panel_x + 20, y_offset))
            y_offset += 18
            shown_count += 1
    
    if shown_count == 0 and not perturbed_body:
        no_alert = font_small.render("Sistema estable - Sin amenazas detectadas", True, (100, 255, 100))
        screen.blit(no_alert, (panel_x + 15, y_offset))
    
    return collision_detected

def draw_doppler_visualization(screen, analyzers, scale, offset, perturbed_body, earth):
    collision_pairs = []
    
    for analyzer in analyzers:
        b1, b2 = analyzer.body1, analyzer.body2
        
        if perturbed_body and (b1 == perturbed_body or b2 == perturbed_body):
            p_earth = earth.pos * scale + offset
            p_target = perturbed_body.pos * scale + offset
            pygame.draw.line(screen, (50, 150, 255), p_earth.astype(int), p_target.astype(int), 1)
        
        p1 = b1.pos * scale + offset
        p2 = b2.pos * scale + offset
        
        doppler_b1, doppler_b2, vrad_b1, vrad_b2, dist, warning = analyzer.calculate_doppler_shift()
        warning_level = analyzer.get_warning_level()
        
        vel_rel = b2.vel - b1.vel
        diff = b2.pos - b1.pos
        if dist > 0:
            radial_vel = np.dot(vel_rel, diff/dist)
        else:
            radial_vel = 0
        
        should_draw = False
        
        if warning_level >= 2:
            should_draw = True
        elif perturbed_body and (b1 == perturbed_body or b2 == perturbed_body):
            if radial_vel < -0.8 and dist < 100:
                should_draw = True
                warning_level = 1
        
        if should_draw:
            collision_pairs.append((b1, b2, warning_level))
            
            if warning_level == 3:
                color = (255, 50, 50)
                width = 5
            elif warning_level == 2:
                color = (255, 150, 0)
                width = 4
            else:
                color = (255, 255, 0)
                width = 2
            
            pygame.draw.line(screen, color, p1.astype(int), p2.astype(int), width)
            
            if warning_level >= 2:
                pulse_radius = int(15 + 8 * abs(math.sin(pygame.time.get_ticks() / 150)))
                pygame.draw.circle(screen, color, p1.astype(int), pulse_radius, 3)
                pygame.draw.circle(screen, color, p2.astype(int), pulse_radius, 3)
                
            if warning_level == 3:
                mid_x = int((p1[0] + p2[0]) / 2)
                mid_y = int((p1[1] + p2[1]) / 2)
                font_alert = pygame.font.SysFont("Arial", 16, bold=True)
                alert_text = font_alert.render("¡COLISIÓN!", True, (255, 255, 255))
                text_rect = alert_text.get_rect(center=(mid_x, mid_y))
                pygame.draw.rect(screen, (255, 0, 0), text_rect.inflate(10, 5))
                screen.blit(alert_text, text_rect)
    
    return collision_pairs

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Sistema Doppler de Detección de Colisiones")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16, bold=True)
    font_small = pygame.font.SysFont("Consolas", 12)
    font_ui = pygame.font.SysFont("Arial", 14)

    scale = 0.9
    offset = np.array([WIDTH//3, HEIGHT//2], dtype=float)
    
    bodies = [
        Body("Sol", SOLAR_MASS, 0, (255, 200, 0), 20, is_sun=True),
        Body("Mercurio", 2,   70,  (160, 160, 160), 4),
        Body("Venus",    3,   110, (200, 180, 100), 6),
        Body("Tierra",   3,   150, (100, 150, 255), 7),
        Body("Marte",    2,   200, (255, 80, 80),   5),
        Body("Ceres",    1,   270, (180, 140, 100), 3),
        Body("Júpiter",  10,  350, (220, 180, 120), 14),
        Body("Saturno",  8,   450, (230, 230, 160), 12),
        Body("Urano",    5,   550, (100, 240, 240), 9),
        Body("Neptuno",  5,   630, (50, 100, 255),  9),
        Body("Plutón",   1,   700, (200, 180, 180), 3),
        Body("Estación", 1,   300, (100, 255, 100), 5, is_station=True),
    ]
    
    earth = next((b for b in bodies if b.name == "Tierra"), None)
    if not earth:
        earth = bodies[3]
    
    analyzers = []
    planets = [b for b in bodies if not b.is_sun and b != earth]
    for i in range(len(planets)):
        for j in range(i+1, len(planets)):
            analyzers.append(DopplerAnalyzer(earth, planets[i], planets[j]))

    running = True
    paused = False
    show_orbits = True
    show_predictions = True
    frame_count = 0
    perturbation_applied = False
    asteroid_pos = None
    collision_pairs = []
    target_planet = None
    
    perturbable_planets = [b for b in bodies if not b.is_sun and not b.is_station and b.name != "Tierra"]
    
    planet_keys = {}
    for i, planet in enumerate(perturbable_planets):
        if i < 9:
            planet_keys[pygame.K_F1 + i] = planet

    while running:
        screen.fill(BG_COLOR)
        mx, my = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEWHEEL:
                if event.y > 0: scale *= 1.1
                else: scale /= 1.1
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    main()
                    return
                elif event.key == pygame.K_o:
                    show_orbits = not show_orbits
                elif event.key == pygame.K_p:
                    show_predictions = not show_predictions
                elif event.key in planet_keys:
                    target_planet = planet_keys[event.key]
                    perturbation_applied = False
                    asteroid_pos = None
                elif target_planet is not None:
                    if event.key == pygame.K_1:
                        nearby = min([b for b in bodies if b != target_planet and not b.is_sun and not b.is_station and b.name != "Tierra"],
                                   key=lambda x: np.linalg.norm(x.pos - target_planet.pos))
                        asteroid_pos = configure_collision_trajectory(target_planet, nearby, 0.3)
                        perturbation_applied = True
                    elif event.key == pygame.K_2:
                        nearby = min([b for b in bodies if b != target_planet and not b.is_sun and not b.is_station and b.name != "Tierra"],
                                   key=lambda x: np.linalg.norm(x.pos - target_planet.pos))
                        asteroid_pos = configure_collision_trajectory(target_planet, nearby, 0.6)
                        perturbation_applied = True
                    elif event.key == pygame.K_3:
                        nearby = min([b for b in bodies if b != target_planet and not b.is_sun and not b.is_station and b.name != "Tierra"],
                                   key=lambda x: np.linalg.norm(x.pos - target_planet.pos))
                        asteroid_pos = configure_collision_trajectory(target_planet, nearby, 0.9)
                        perturbation_applied = True
                    elif event.key == pygame.K_a:
                        asteroid_pos = apply_gravitational_perturbation(target_planet, bodies)
                        perturbation_applied = True

        if not paused:
            frame_count += 1
            
            for b in bodies:
                b.update_physics(bodies)
            
            if perturbation_applied and target_planet and frame_count % 60 == 0:
                for analyzer in analyzers:
                    if analyzer.body1 == target_planet or analyzer.body2 == target_planet:
                        will_collide, collision_point, collision_frame = analyzer.predict_collision_using_doppler(bodies)
                        if will_collide:
                            analyzer.collision_predicted = True
                            analyzer.predicted_collision_point = collision_point
                            analyzer.collision_frame = collision_frame
                        else:
                            analyzer.collision_predicted = False
            
            if perturbation_applied and target_planet and frame_count % 30 == 0:
                if show_predictions:
                    target_planet.future_positions = target_planet.predict_trajectory(bodies)
                    for pair in collision_pairs:
                        if len(pair) >= 2:
                            pair[0].future_positions = pair[0].predict_trajectory(bodies, 40)
                            pair[1].future_positions = pair[1].predict_trajectory(bodies, 40)
            
            for b in bodies:
                b.update_physics(bodies)
            
            if perturbation_applied and frame_count % 30 == 0:
                if show_predictions:
                    target_planet.future_positions = target_planet.predict_trajectory(bodies)
                    for pair in collision_pairs:
                        if len(pair) >= 2:
                            pair[0].future_positions = pair[0].predict_trajectory(bodies, 40)
                            pair[1].future_positions = pair[1].predict_trajectory(bodies, 40)

        if show_orbits:
            for b in bodies:
                b.draw_orbit_guide(screen, scale, offset)
        
        collision_pairs = draw_doppler_visualization(screen, analyzers, scale, offset, target_planet if perturbation_applied else None, earth)
        
        if asteroid_pos is not None and perturbation_applied and frame_count < 180:
            ax = int(asteroid_pos[0] * scale + offset[0])
            ay = int(asteroid_pos[1] * scale + offset[1])
            pygame.draw.circle(screen, (150, 150, 150), (ax, ay), 8)
            pygame.draw.circle(screen, (255, 255, 255), (ax, ay), 12, 2)
        
        for b in bodies:
            is_selected = (b == target_planet and not perturbation_applied)
            b.draw(screen, scale, offset, is_selected)

        collision_detected = draw_doppler_panel(screen, analyzers, font, font_small, 
                                               target_planet if perturbation_applied else None, earth)
        
        if target_planet is None:
            status_text = "Selecciona un planeta (F1-F9)"
            status_color = (150, 200, 255)
        elif not perturbation_applied:
            status_text = f"✓ {target_planet.name.upper()} SELECCIONADO - Presiona 1, 2, 3 o A"
            status_color = (0, 255, 200)
        else:
            status_text = f"{target_planet.name.upper()} PERTURBADO"
            status_color = (255, 200, 50)
        
        ui_lines = [
            f"ESTADO: {status_text}",
            "",
            "SELECCIONAR PLANETA:",
            "• F1: Mercurio   F2: Venus   F3: Marte",
            "• F4: Ceres      F5: Júpiter F6: Saturno",
            "• F7: Urano      F8: Neptuno F9: Plutón",
            "",
            "ESCENARIOS DE PERTURBACIÓN:",
            "• 1-3: Asistencia Gravitacional (encuentro cercano)",
            "  └─ Leve/Media/Fuerte según distancia de paso",
            "• A: Objeto Interestelar (asteroide/planeta errante)",
            "",
            "CONTROLES:",
            "• ESPACIO: Pausar    R: Reiniciar",
            "• O: Órbitas    P: Predicciones"
        ]
        
        y_ui = HEIGHT - 320
        for i, line in enumerate(ui_lines):
            if i == 0:
                color = status_color
            elif "SELECCIONAR" in line or "ESCENARIOS" in line or "CONTROLES" in line:
                color = (150, 200, 255)
            elif "└─" in line:
                color = (150, 150, 150)
            else:
                color = (180, 180, 180)
            ui_text = font_ui.render(line, True, color)
            screen.blit(ui_text, (10, y_ui + i * 20))
        
        if target_planet:
            if perturbation_applied:
                status_bg = pygame.Surface((320, 40))
                status_bg.set_alpha(220)
                status_bg.fill((50, 30, 10))
                screen.blit(status_bg, (WIDTH//2 - 160, 5))
                
                status_text = font.render(f"📡 Monitoreando: {target_planet.name}", True, (255, 150, 50))
                screen.blit(status_text, (WIDTH//2 - 150, 12))
            else:
                status_bg = pygame.Surface((360, 40))
                status_bg.set_alpha(200)
                status_bg.fill((30, 50, 30))
                screen.blit(status_bg, (WIDTH//2 - 180, 5))
                
                status_text = font.render(f"✓ {target_planet.name} seleccionado - Presiona 1,2,3 o A", True, (100, 255, 100))
                screen.blit(status_text, (WIDTH//2 - 170, 12))
        else:
            status_bg = pygame.Surface((400, 40))
            status_bg.set_alpha(200)
            status_bg.fill((30, 30, 50))
            screen.blit(status_bg, (WIDTH//2 - 200, 5))
            
            status_text = font.render("Sistema Estable - Presiona F1-F9 para seleccionar planeta", True, (150, 150, 255))
            screen.blit(status_text, (WIDTH//2 - 190, 12))
        
        if paused:
            pause_text = font.render("|| PAUSADO ||", True, (255, 255, 0))
            screen.blit(pause_text, (WIDTH//2 - 70, 40))
        
        if collision_detected:
            max_warning = max(analyzer.get_warning_level() for analyzer in analyzers)
            if max_warning == 3:
                alert_text = font.render("⚠⚠⚠ COLISIÓN INMINENTE DETECTADA POR DOPPLER ⚠⚠⚠", True, (255, 50, 50))
                pulse_alpha = int(100 + 155 * abs(math.sin(pygame.time.get_ticks() / 200)))
                alert_bg = pygame.Surface((WIDTH, 40))
                alert_bg.set_alpha(pulse_alpha)
                alert_bg.fill((100, 0, 0))
                screen.blit(alert_bg, (0, HEIGHT - 40))
            else:
                alert_text = font.render("⚠ ALERTA: TRAYECTORIA DE COLISIÓN DETECTADA POR DOPPLER ⚠", True, (255, 150, 0))
            screen.blit(alert_text, (WIDTH//2 - 350, HEIGHT - 30))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()