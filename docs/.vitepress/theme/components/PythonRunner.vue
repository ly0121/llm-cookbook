<template>
  <ClientOnly>
    <div class="python-runner">
      <div class="python-runner__header">
        <span>🐍 Python {{ browserRunnable ? '(浏览器运行)' : '(需要本地环境)' }}</span>
        <div class="python-runner__actions">
          <button
            v-if="browserRunnable"
            class="python-runner__btn python-runner__btn--run"
            :disabled="running || loading"
            @click="run"
          >
            {{ loading ? '⏳ 加载 Pyodide...' : running ? '⏳ 运行中...' : '▶ 运行' }}
          </button>
          <button
            v-if="browserRunnable && edited"
            class="python-runner__btn python-runner__btn--reset"
            @click="reset"
          >
            ↺ 重置
          </button>
        </div>
      </div>
      <textarea
        class="python-runner__editor"
        v-model="currentCode"
        :readonly="!browserRunnable"
        :rows="codeLines"
        spellcheck="false"
      ></textarea>
      <div
        v-if="output || error || presetOutput"
        class="python-runner__output"
        :class="{ 'python-runner__output--error': !!error }"
      >{{ error || output || presetOutput }}</div>
      <div v-if="executionTime" class="python-runner__status">
        ⏱️ 执行耗时: {{ executionTime }}ms
      </div>
    </div>
  </ClientOnly>
</template>

<script setup lang="ts">
import { ref, computed, useSlots, onMounted } from 'vue'

const props = defineProps<{
  code?: string
  browserRunnable?: boolean
  presetOutput?: string
}>()

const slots = useSlots()

function getSlotText(): string {
  if (props.code) return props.code
  const slot = slots.default?.()
  if (!slot || !slot.length) return ''
  // Extract text from slot VNodes
  return slot.map((vnode: any) => {
    if (typeof vnode.children === 'string') return vnode.children
    return ''
  }).join('')
}

const initialCode = ref('')

onMounted(() => {
  initialCode.value = props.code || getSlotText() || ''
  currentCode.value = initialCode.value
})

const currentCode = ref('')
const output = ref('')
const error = ref('')
const running = ref(false)
const loading = ref(false)
const executionTime = ref<number | null>(null)

const edited = computed(() => currentCode.value !== initialCode.value)
const codeLines = computed(() => Math.max(4, (currentCode.value || '').split('\n').length + 1))

async function run() {
  if (!props.browserRunnable) return

  running.value = true
  output.value = ''
  error.value = ''
  executionTime.value = null

  try {
    loading.value = true
    const { runPython } = await import('../../utils/pyodide')
    loading.value = false

    const start = performance.now()
    const result = await runPython(currentCode.value)
    executionTime.value = Math.round(performance.now() - start)

    output.value = result.output
    error.value = result.error
  } catch (e: any) {
    error.value = `加载失败: ${e.message}`
    loading.value = false
  } finally {
    running.value = false
  }
}

function reset() {
  currentCode.value = initialCode.value
  output.value = ''
  error.value = ''
  executionTime.value = null
}
</script>
