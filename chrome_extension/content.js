console.log("CONTENT SCRIPT LOADED") //

// SHA-256 해시 기반 UUID 생성 (UTF-8)
async function generateUUID(text) {
    const encoder = new TextEncoder(); // 문자열을 UTF-8 바이트로 인코딩하는 인코더 생성
    const data = encoder.encode(text); // 텍스트를 바이트 배열로 변환
    const hashBuffer = await crypto.subtle.digest('SHA-256', data); // SHA-256 해시값 생성 (비동기)
    const hashArray = Array.from(new Uint8Array(hashBuffer)); // 버퍼를 배열로 변환
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join(""); // 16진수 문자열로 조합(UUID처럼 사용)
}

// 서버에 댓글을 전송하고 결과에 따라 blur 처리
async function processCommentsWithServer(comments) {
    // UUID 매핑 객체 생성
    const payload = await Promise.all(comments.map(async ({ element, text }) => {
        const uuid = await generateUUID(text);
        return { id: uuid, text } // { id: uuid, text: text }
    }));

    console.log("[DEBUG] 서버로 보낼 payload:", payload); //

    // 서버(Django API)로 댓글 ID 및 내용을 전송하여 예측 결과를 요청
    const response = await fetch(`${SERVER_URL}/check_comments/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });

    const results = await response.json(); // [{ id, prob, predict }, ...]
    console.log("[DEBUG] 서버 응답:", results); //

    // 결과 매핑 및 처리
    for (const { id, prob, predict } of results) {
        for (const comment of comments) {
            const uuid = await generateUUID(comment.text);
            if (uuid === id && predict === 'ABUSE') {
                const {element} = comment;

                // blur 효과 적용
                element.style.filter = 'blur(3px)';
                element.style.position = 'relative';
                element.style.cursor = 'pointer';

                // 툴팁 생성
                const tooltip = document.createElement('div');
                tooltip.textContent = `악성 확률: ${(prob * 100).toFixed(2)}%`;
                tooltip.className = 'abuse-tooltip';

                // 확률에 따라 배경색 설정
                if (prob >= 0.9) { // 0.9 이상은 고도
                    tooltip.style.background = 'rgba(255, 0, 0, 0.85)'; // 빨간색
                    tooltip.style.color = '#fff'; // 하얀 글씨
                } else if (prob >= 0.8) { // 0.8 이상은 중등도
                    tooltip.style.background = 'rgba(255, 140, 0, 0.85)'; // 주황색
                } else { // 나머지는 경도
                    tooltip.style.background = 'rgba(255, 215, 0, 0.85)'; // 노란색
                }

                // 위치 초기화 (안 보이지 않게 하기 위해)
                tooltip.style.left = '0px';
                tooltip.style.top = '0px';

                // DOM에 추가
                document.body.appendChild(tooltip);
                console.log("툴팁 DOM 추가:", tooltip); // 툴팁 추가 확인

                // 마우스 위치에 따라 툴팁 이동
                const moveTooltip = (e) => {
                    console.log('[DEBUG] mousemove 발생:', e.clientX, e.clientY);

                    // 툴팁이 화면을 벗어나지 않도록 위치 조정
                    const tooltipWidth = tooltip.offsetWidth;
                    const tooltipHeight = tooltip.offsetHeight;
                    const maxX = window.innerWidth - tooltipWidth - 10;
                    const maxY = window.innerHeight - tooltipHeight - 10;

                    let x = e.clientX + 10;
                    let y = e.clientY + 10;

                    // 화면 크기 안에서 툴팁이 보이도록 조정
                    x = Math.max(0, Math.min(x, maxX));
                    y = Math.max(0, Math.min(y, maxY));

                    tooltip.style.left = `${x}px`;
                    tooltip.style.top = `${y}px`;

                    tooltip.style.opacity = '1'; // 툴팁 보이기
                };

                const hideTooltip = () => {
                    tooltip.style.opacity = '0'; // 툴팁 숨기기
                };

                element.addEventListener('mouseenter', (e) => {
                    // 툴팁 위치 초기화 및 표시
                    tooltip.style.position = 'fixed';
                    tooltip.style.opacity = '1';
                    tooltip.style.display = 'block'; // 혹시 display 문제 있을 가능성 예방

                    moveTooltip(e); // 마우스 올렸을 때 바로 툴팁 위치 지정
                });
                element.addEventListener('mousemove', moveTooltip);
                element.addEventListener('mouseleave', hideTooltip);

                // 클릭 시 blur 해제 및 툴팁 제거
                element.addEventListener('click', () => {
                    element.style.filter = 'none';
                    tooltip.remove();
                    element.removeEventListener('mousemove', moveTooltip);
                    element.removeEventListener('mouseleave', hideTooltip);
                });

                console.log(`처리 완료: "${comment.text.substring(0, 30)}"... | 예측: ${predict} | 악성 확률: (${(prob * 100).toFixed(2)}%)`);
            }
        }
    }
}

// 댓글 수집 및 처리
async function startCommentProcessing() {
    console.log("[DEBUG] startCommentProcessing() 호출됨"); //

    const commentElements = Array.from(document.querySelectorAll('#content-text'));
    console.log("탐지된 댓글 수:", commentElements.length); //

    // 이미 처리된 댓글을 구분하기 위해 Set 사용
    const newComments = commentElements
        .filter(el => !el.dataset.processed)
        .map(el => ({ element: el, text: el.innerText }));

    if (newComments.length === 0) return;

    // 이미 처리된 댓글로 마킹
    newComments.forEach(({ element }) => {
        element.dataset.processed = 'true';
    });

    // 서버에 새 댓글 전송
    await processCommentsWithServer(newComments);
}

// 댓글 영역 로딩 감지
function waitForCommentContainer(callback) {
    const interval = setInterval(() => {
        const container = document.querySelector('#contents');
        if (container) {
            clearInterval(interval);
            callback(container);
        }
    }, 300);
}

function isVideoPage() {
    // YouTube 동영상 페이지는 "/watch?v=" 경로를 포함.
    return location.href.includes("/watch?v=");
}

// URL 변경 감지
let lastUrl = location.href;
new MutationObserver(() => {
    const currentUrl = location.href;
    if (currentUrl !== lastUrl) {
        lastUrl = currentUrl;
        console.log(`[URL 변경 감지] ${currentUrl}`); //
        if (isVideoPage()) { // 영상 페이지일 때만
            waitForCommentContainer((container) => {
                observeNewComments(container);
                startCommentProcessing(); // 페이지 이동 시 댓글 처리 재시작
            });
        }
    }
}).observe(document, { subtree: true, childList: true });

// 댓글 추가 감지 + 스크롤 fallback
function observeNewComments(container) {
    const observer = new MutationObserver(() => {
        console.log('[DEBUG] 새 댓글 감지됨'); //
        startCommentProcessing(); // 무조건 실행
    });

    observer.observe(container, { childList: true, subtree: true });

    // 스크롤 이벤트로 fallback 처리
    window.addEventListener('scroll', () => {
        startCommentProcessing();
    });
}

// 초기 실행
waitForCommentContainer((container) => {
    if (isVideoPage()) {
        observeNewComments(container);
        startCommentProcessing();
    }
});