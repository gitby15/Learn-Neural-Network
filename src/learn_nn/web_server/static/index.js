
console.log('hello world')
async function get_pos_data() {
    api_path = '/api/cross_product_pos_embedding'
    const response = await fetch(api_path)
    const data = await response.json()
    // [B, T, C] 
    console.log('list data: ', data)
    return data
}



async function draw_tensor(first_batch) {
    // const data = await get_pos_data() // [B, T, C]
    // const first_batch = data.pos[0]

    // 取B,C维度, 画一张二维表格，垂直方向表示T的位置，水平方向表示C的位置，表格的颜色用rgb来表达C的值
    // 然后用canvas画出来
    const canvas = document.createElement('canvas')
    const t_size = first_batch.length
    const c_size = first_batch[0].length
    const scale_size = 2
    // 横向表示位置编码的值
    canvas.width = c_size * scale_size
    // 纵向标志位置
    canvas.height = t_size * scale_size
    const ctx = canvas.getContext('2d')
    ctx.strokeStyle = 'black'
    ctx.lineWidth = 2
    ctx.strokeRect(0, 0, canvas.width, canvas.height)
    for (let t_step = 0; t_step < t_size; t_step++) {
        const pixel_list = []
        for (let pos_step = 0; pos_step < c_size; pos_step++) {
            // 值的范围是[-1,1]
            let pixel_value = first_batch[t_step][pos_step]
            // 用 hue(色相) 编码数值大小，lightness(亮度) 编码是否接近 0
            const t = (pixel_value + 1) / 2        // [0, 1]
            const hue = 240 * (1 - t)              // 红(0) → 蓝(240)，360° 色相全用
            const lightness = 30 + 40 * t          // 30%~70% 亮度，避免纯黑纯白
            ctx.fillStyle = `hsl(${hue}, 100%, ${lightness}%)`
            ctx.fillRect(pos_step * scale_size, t_step * scale_size, scale_size, scale_size)
        }
        console.log(pixel_list)
    }
    document.body.appendChild(canvas)
}

async function draw_pos_embedding(){
    data = await get_pos_data()
    draw_tensor(data.embed[0])
    draw_tensor(data.pos_zero[0])
    draw_tensor(data.pos_embed[0])
    

}

draw_pos_embedding()
