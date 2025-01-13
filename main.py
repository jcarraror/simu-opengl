import math
import random
from typing import List, Tuple, Dict, Optional

import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

import spacy
from PIL import Image, ImageDraw, ImageFont

# Carregando o modelo
nlp = spacy.load("pt_core_news_sm")

########################################
# CONFIGURAÇÕES GERAIS
########################################
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FPS = 60
MAX_LIGHT_RAYS = 20

# Mapear POS conhecidas em cores específicas
POS_COLORS = {
    "VERB": (1.0, 0.3, 0.3),
    "NOUN": (0.3, 1.0, 0.3),
    "ADJ":  (0.3, 0.3, 1.0),
    "ADV":  (1.0, 1.0, 0.3),
    # Incluir mais classes
    #TODO: Melhorar o mapeamento de cores (mais classes e de uma forma mais dinâmica e parametrizada)
}

########################################
# SHADERS (GLSL)
########################################

VERTEX_SHADER_CODE = r"""
#version 330 core

layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;

uniform mat4 uProjection;
uniform vec2 uTranslation;
uniform float uScale;

out vec2 vTexCoord;

void main()
{
    vec2 scaledPos = aPos * uScale + uTranslation;
    vTexCoord = aTexCoord;
    gl_Position = uProjection * vec4(scaledPos, 0.0, 1.0);
}
""";

FRAGMENT_SHADER_CODE = r"""
#version 330 core

in vec2 vTexCoord;
out vec4 FragColor;

uniform vec3 uColor;
uniform float uAlpha;

void main()
{
    // Distância ao centro do quad
    float dx = vTexCoord.x - 0.5;
    float dy = vTexCoord.y - 0.5;
    float dist = sqrt(dx*dx + dy*dy);

    // "glow" radial
    float glow = 1.0 - dist * 2.0; 
    if(glow < 0.0) {
        glow = 0.0;
    }

    // Cor final
    FragColor = vec4(uColor, glow * uAlpha);
}
""";

########################################
# FUNÇÕES AUXILIARES (SHADER, VAO, ETC.)
########################################
import ctypes

def compile_shader(source, shader_type):
    '''Compila um shader GLSL e retorna o ID do shader'''
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(shader).decode()
        raise RuntimeError(f"Erro compilando shader: {error}")
    return shader

def create_shader_program(vs_code, fs_code):
    '''Cria um programa de shader GLSL e retorna o ID do programa'''
    vs = compile_shader(vs_code, GL_VERTEX_SHADER)
    fs = compile_shader(fs_code, GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vs)
    glAttachShader(program, fs)
    glLinkProgram(program)
    if not glGetProgramiv(program, GL_LINK_STATUS):
        error = glGetProgramInfoLog(program).decode()
        raise RuntimeError(f"Erro linkando programa: {error}")
    glDeleteShader(vs)
    glDeleteShader(fs)
    return program

def create_quad_vao() -> int:
    '''Cria um VAO para um quadrado unitário (pra desenhar partículas)'''
    #VAO -> Vertex Array Object
    vertices = np.array([
        # x     y     u     v
        -0.5, -0.5,  0.0,  0.0,
         0.5, -0.5,  1.0,  0.0,
         0.5,  0.5,  1.0,  1.0,
        -0.5,  0.5,  0.0,  1.0,
    ], dtype=np.float32) # 4 vértices

    indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32) # 2 triângulos

    VAO = glGenVertexArrays(1) # ID do VAO
    VBO = glGenBuffers(1) # ID do VBO (Vertex Buffer Object)
    EBO = glGenBuffers(1) # ID do EBO (Element Buffer Object)

    glBindVertexArray(VAO) # Ativa o VAO

    glBindBuffer(GL_ARRAY_BUFFER, VBO) # Ativa o VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW) # Carrega os dados

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO) # Ativa o EBO
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW) # Carrega os dados

    # Atributo 0 -> aPos
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4*4, ctypes.c_void_p(0)) # 4*4 = 16 bytes
    glEnableVertexAttribArray(0) # Ativa o atributo 0

    # Atributo 1 -> aTexCoord
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4*4, ctypes.c_void_p(8)) # 4*4 = 16 bytes
    glEnableVertexAttribArray(1) # Ativa o atributo 1

    glBindVertexArray(0) # Desativa o VAO
    return VAO

def create_text_texture(text: str, font_size=18) -> Tuple[int, int, int]:
    '''Cria uma textura com o texto informado'''
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()

    dummy_img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0,0), text, font=font) # bbox = (x0, y0, x1, y1)
    w = bbox[2] - bbox[0] # largura (x1 - x0)
    h = bbox[3] - bbox[1] # altura (y1 - y0)

    if w <= 0:  # evitar caso de texto vazio
        w = 1
    if h <= 0:
        h = 1

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0)) # imagem preta transparente
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font, fill=(255, 255, 255, 255)) # texto branco

    raw_data = img.tobytes("raw", "RGBA", 0, -1) # raw_data = bytes
    tex_id = glGenTextures(1) # ID da textura
    glBindTexture(GL_TEXTURE_2D, tex_id) # Ativa a textura

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE) # Borda
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE) # Borda
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR) # Filtro
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR) # Filtro

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, raw_data) # Carrega a textura
    glBindTexture(GL_TEXTURE_2D, 0) # Desativa a textura

    return tex_id, w, h

########################################
# ANÁLISE DE TEXTO
########################################
def analyze_text(text: str):
    ''' Analisa o texto e retorna dados sobre os tokens e sentimento '''
    doc = nlp(text) # Processa o texto
    tokens_data = []
    for token in doc:
        # Exemplo: (palavra, POS)
        tokens_data.append((token.text, token.pos_)) # POS -> Part of Speech

    # Exemplo simples de "sentimento aleatório"
    return {
        "tokens_data": tokens_data,
        "total_words": len(doc),
        "sentiment": random.uniform(-1.0, 1.0) #TODO: Melhorar a análise de sentimento (usar um modelo real)
    }

def pos_to_color(pos: str) -> Tuple[float, float, float]:
    '''Converte uma tag POS para uma cor RGB (com fallback)'''
    # Normaliza para maiúsculo
    pos_upper = pos.upper()
    if pos_upper in POS_COLORS:
        return POS_COLORS[pos_upper]
    # Tenta minúsculo
    pos_lower = pos.lower()
    if pos_lower in POS_COLORS:
        return POS_COLORS[pos_lower]
    # Senão, cor aleatória
    return (random.random(), random.random(), random.random())

########################################
# CLASSE DO FEIXE DE LUZ
########################################
class LightRay:
    ''' Representa um feixe de luz com uma palavra e uma posição gramatical '''
    def __init__(self, x, y, angle, speed, color, radius, word, pos):
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = speed
        self.color = color  # (r, g, b)
        self.radius = radius
        self.word = word
        self.pos = pos

    def update(self, dt: float):
        '''Atualiza a posição do feixe de luz'''
        self.x += math.cos(self.angle) * self.speed * dt # x = x + cos(ang) * vel * dt
        self.y += math.sin(self.angle) * self.speed * dt # y = y + sin(ang) * vel * dt

        # Reflexão
        if self.x < 0 or self.x > WINDOW_WIDTH:
            self.angle = math.pi - self.angle # ang = pi - ang
        if self.y < 0 or self.y > WINDOW_HEIGHT:
            self.angle = -self.angle # ang = -ang

    def is_mouse_hover(self, mx, my) -> bool:
        '''Verifica se o mouse está em cima do feixe'''
        dist = math.dist((self.x, self.y), (mx, my)) # distância euclidiana
        return dist < self.radius / 2

    def get_tooltip_text(self):
        '''Retorna o texto do tooltip para o feixe'''
        return f"{self.word} ({self.pos})\nVelocidade={self.speed:.1f}"

########################################
# CRIAR FEIXES A PARTIR DO TEXTO
########################################
def create_light_rays_from_text(text: str) -> List[LightRay]:
    """
    Cria feixes de luz a partir do texto inserido, sem repetir tokens.
    Se houver mais tokens que o limite (MAX_LIGHT_RAYS), usaremos apenas
    os primeiros (depois de embaralhar).
    """

    analysis = analyze_text(text) # Dados da análise
    data = analysis["tokens_data"]  # lista de (palavra, POS)
    total = analysis["total_words"] # total de palavras
    sentiment = analysis["sentiment"] # sentimento

    # Se não tiver tokens, criamos pelo menos 1 para não travar
    if total < 1:
        total = 1

    # Número efetivo de partículas = min(total, MAX_LIGHT_RAYS)
    num_rays = min(total, MAX_LIGHT_RAYS)

    # Embaralha a lista de tokens para evitar sempre pegar os primeiros
    random.shuffle(data)

    # Pegamos apenas o 'num_rays' primeiros tokens, sem repetição
    used_tokens = data[:num_rays]

    new_rays = []
    for (w, p) in used_tokens:
        # Ângulo aleatório + leve ajuste pelo "sentiment"
        angle = random.uniform(0, math.pi * 2)
        angle += (sentiment * 0.5)

        # Velocidade base
        speed = 50.0
        if p.upper() == "VERB":
            speed += 150 #TODO: Adicionar comportamento de velocidade para outras classes e determinar velocidade a partir de melhores critérios (como tamanho da palavra, relação com outras palavras, etc.)

        color = pos_to_color(p) # Cor baseada na POS

        # Tamanho maior ou menor do glow
        radius = random.uniform(50, 70) #TODO: Melhorar o cálculo do raio (usar tamanho da palavra, por exemplo)

        # Posição inicial aleatória na janela
        x = random.uniform(0, WINDOW_WIDTH)
        y = random.uniform(0, WINDOW_HEIGHT)

        new_rays.append(LightRay(x, y, angle, speed, color, radius, w, p))

        #TODO: Adicionar mais comportamentos de partícula com base na análise de texto (diferentes tamanhos, cores, velocidades, animações, interações, etc.)

    return new_rays

########################################
# TOOLTIP
########################################
class TooltipManager:
    '''Gerencia a exibição de tooltips na tela'''
    def __init__(self, show_tooltip=True):
        self.show_tooltip = show_tooltip
        self.cache = {}

    def clear_cache(self):
        '''Limpa cache das texturas (evitar info desatualizada)'''
        self.cache.clear()

    def get_text_texture(self, text: str, font_size=14):
        '''Retorna a textura de um texto'''
        if text in self.cache:
            return self.cache[text]
        tid, w, h = create_text_texture(text, font_size=font_size)
        self.cache[text] = (tid, w, h)
        return tid, w, h

    def draw_tooltip(self, text: str, x: float, y: float):
        '''Desenha um tooltip na tela'''
        if not self.show_tooltip or not text:
            return

        tid, w, h = self.get_text_texture(text)
        x_off, y_off = 10, 10
        margin = 4

        left   = x + x_off
        right  = left + w + margin*2
        bottom = y + y_off
        top    = bottom + h + margin*2

        # fundo
        glDisable(GL_TEXTURE_2D) # desativa textura
        glEnable(GL_BLEND) # ativa blend
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) # blend normal
        glColor4f(0.0, 0.0, 0.0, 0.6) # cor do fundo

        glBegin(GL_QUADS) # desenha um quadrado
        glVertex2f(left,  bottom) # vértice 1
        glVertex2f(right, bottom) # vértice 2
        glVertex2f(right, top)   # vértice 3
        glVertex2f(left,  top)  # vértice 4
        glEnd() # finaliza

        # texto
        glEnable(GL_TEXTURE_2D) # ativa textura
        glBindTexture(GL_TEXTURE_2D, tid) # ativa a textura
        glColor4f(1,1,1,1) # cor do texto

        glBegin(GL_QUADS) # desenha um quadrado
        glTexCoord2f(0,0); glVertex2f(left+margin,  bottom+margin) # vértice 1
        glTexCoord2f(1,0); glVertex2f(left+margin+w, bottom+margin) # vértice 2
        glTexCoord2f(1,1); glVertex2f(left+margin+w, bottom+margin+h) # vértice 3
        glTexCoord2f(0,1); glVertex2f(left+margin,  bottom+margin+h) # vértice 4
        glEnd() # finaliza

        glBindTexture(GL_TEXTURE_2D, 0) # desativa a textura
        glDisable(GL_TEXTURE_2D) # desativa textura

########################################
# DESENHAR UMA LINHA DE TEXTO (INPUT)
########################################
class TextRenderer:
    '''Renderiza texto na tela'''
    def __init__(self):
        self.cache = {}

    def clear_cache(self):
        """Se precisar remover entradas antigas"""
        self.cache.clear()

    def draw_text(self, text: str, x: float, y: float, font_size=18):
        '''Desenha um texto na tela'''
        if text not in self.cache:
            tex_id, w, h = create_text_texture(text, font_size)
            self.cache[text] = (tex_id, w, h)

        tex_id, w, h = self.cache[text]

        # retângulo de destino
        left   = x
        right  = x + w
        bottom = y
        top    = y + h

        glEnable(GL_TEXTURE_2D) # ativa textura
        glBindTexture(GL_TEXTURE_2D, tex_id) # ativa a textura
        glColor4f(1,1,1,1) # cor do texto
        glBegin(GL_QUADS) # desenha um quadrado
        glTexCoord2f(0,0); glVertex2f(left,  bottom) # vértice 1
        glTexCoord2f(1,0); glVertex2f(right, bottom) # vértice 2
        glTexCoord2f(1,1); glVertex2f(right, top)  # vértice 3
        glTexCoord2f(0,1); glVertex2f(left,  top) # vértice 4
        glEnd() # finaliza

        glBindTexture(GL_TEXTURE_2D, 0) # desativa a textura
        glDisable(GL_TEXTURE_2D) # desativa textura

########################################
# CALLBACKS DE TECLADO
########################################
typing_buffer = []  # guarda caracteres digitados

def char_callback(window, codepoint):
    '''Callback para tratar caracteres digitados'''
    global typing_buffer
    ch = chr(codepoint) # converte para caractere
    typing_buffer.append(ch) # adiciona ao buffer

def key_callback(window, key, scancode, action, mods):
    '''Callback para tratar teclas pressionadas'''
    global typing_buffer, rays

    if action == glfw.PRESS:
        if key == glfw.KEY_BACKSPACE:
            if typing_buffer:
                typing_buffer.pop()
        elif key == glfw.KEY_ENTER:
            # Monta a string final e gera novos feixes
            new_text = "".join(typing_buffer).strip()
            if new_text:
                # Limpa caches de tooltip e texto (para não mostrar info antiga)
                tooltip_manager.clear_cache()
                text_renderer.clear_cache()
                # Gera feixes
                rays = create_light_rays_from_text(new_text)

            typing_buffer = []
        elif key == glfw.KEY_ESCAPE:
            # Pode fechar a janela
            glfw.set_window_should_close(window, True)

########################################
# LOOP PRINCIPAL
########################################

def main_loop():
    global rays, tooltip_manager, text_renderer

    # Texto inicial
    initial_text = "Batatinha quando nasce, esparrama pelo chão. Menininha quando dorme, põe a mão no coração."
    rays = create_light_rays_from_text(initial_text)

    # Init GLFW
    if not glfw.init():
        raise RuntimeError("Falha ao inicializar GLFW.")

    window = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT, "Simu", None, None) # janela
    if not window:
        glfw.terminate()
        raise RuntimeError("Falha ao criar janela GLFW.")

    glfw.make_context_current(window) # contexto

    # Setar callbacks
    glfw.set_char_callback(window, char_callback) # caracteres
    glfw.set_key_callback(window, key_callback) # teclas

    # Projeção ortográfica
    glMatrixMode(GL_PROJECTION) # matriz de projeção
    glLoadIdentity() # carrega matriz identidade
    gluOrtho2D(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT) # define projeção ortográfica
    glMatrixMode(GL_MODELVIEW) # matriz de modelo
    glLoadIdentity() # carrega matriz identidade

    # Compilar shaders
    program = create_shader_program(VERTEX_SHADER_CODE, FRAGMENT_SHADER_CODE) # programa de shader
    loc_uProj = glGetUniformLocation(program, "uProjection") # localização da matriz de projeção
    loc_uTrans= glGetUniformLocation(program, "uTranslation") # localização da translação
    loc_uScale= glGetUniformLocation(program, "uScale") # localização da escala
    loc_uColor= glGetUniformLocation(program, "uColor") # localização da cor 
    loc_uAlpha= glGetUniformLocation(program, "uAlpha") # localização da transparência

    # Matriz Ortho
    left, right = 0.0, float(WINDOW_WIDTH)
    bottom, top = 0.0, float(WINDOW_HEIGHT)
    near, far = -1.0, 1.0
    ortho_mat = np.array([
        [2/(right-left),     0,                 0,                0],
        [0,                  2/(top-bottom),    0,                0],
        [0,                  0,                -2/(far-near),     0],
        [-(right+left)/(right-left),
         -(top+bottom)/(top-bottom),
         -(far+near)/(far-near),
         1]
    ], dtype=np.float32) # matriz 4x4

    quadVAO = create_quad_vao() # VAO do quadrado

    tooltip_manager = TooltipManager(show_tooltip=True) # gerenciador de tooltips
    text_renderer = TextRenderer()

    last_time = glfw.get_time()

    while not glfw.window_should_close(window):
        glfw.poll_events()

        current_time = glfw.get_time()
        dt = current_time - last_time
        last_time = current_time

        # Atualização das partículas
        for r in rays:
            r.update(dt)

        glClearColor(0,0,0,1) # cor de fundo
        glClear(GL_COLOR_BUFFER_BIT) # limpa o buffer de cor

        # Desenhar feixes com blending aditivo
        glUseProgram(program) # ativa o programa de shader
        glUniformMatrix4fv(loc_uProj, 1, GL_FALSE, ortho_mat) # matriz de projeção
        glBindVertexArray(quadVAO) # ativa o VAO

        for r in rays:
            glEnable(GL_BLEND) # ativa blend
            glBlendFunc(GL_SRC_ALPHA, GL_ONE)  # aditivo
            glUniform2f(loc_uTrans, r.x, r.y) # translação
            glUniform1f(loc_uScale, r.radius) # escala
            glUniform3f(loc_uColor, r.color[0], r.color[1], r.color[2]) # cor
            glUniform1f(loc_uAlpha, 1.0) # transparência
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None) # desenha
            # Restaura blend normal
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glBindVertexArray(0) # desativa o VAO
        glUseProgram(0) # desativa o programa de shader

        # Desenhar tooltip
        mx, my = glfw.get_cursor_pos(window)
        my = WINDOW_HEIGHT - my
        for r in rays:
            if r.is_mouse_hover(mx, my):
                tooltip_manager.draw_tooltip(r.get_tooltip_text(), mx, my)
                # break  # Se quiser mostrar apenas o primeiro que encontrar

        # Desenhar linha de input (no topo)
        #TODO: Muito escroto, melhorar isso
        typed_str = "".join(typing_buffer)
        text_line = f"> {typed_str}"
        text_renderer.draw_text(text_line, x=10, y=WINDOW_HEIGHT-25, font_size=20)

        glfw.swap_buffers(window)

    glfw.terminate()

########################################
# EXECUÇÃO
########################################
if __name__ == "__main__":
    main_loop()
