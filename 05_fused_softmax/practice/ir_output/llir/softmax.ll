; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"

@global_smem = external addrspace(3) global [0 x i8], align 16

define void @_softmax_kernel(ptr addrspace(1) %0, ptr addrspace(1) %1, i32 %2, i32 %3, i32 %4, i32 %5) local_unnamed_addr !dbg !7 {
  %7 = tail call i32 asm "mov.u32 $0, %ctaid.x;", "=r"() #3, !dbg !10
  %8 = tail call i32 asm "mov.u32 $0, %nctaid.x;", "=r"() #3, !dbg !11
  %9 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !dbg !12
  %10 = shl i32 %9, 1, !dbg !12
  %11 = and i32 %10, 510, !dbg !12
  %12 = icmp slt i32 %11, %5, !dbg !13
  %13 = icmp slt i32 %7, %4, !dbg !14
  %14 = mul i32 %7, %2, !dbg !15
  %15 = sext i32 %14 to i64, !dbg !16
  %16 = getelementptr float, ptr addrspace(1) %0, i64 %15, !dbg !16
  %17 = zext nneg i32 %11 to i64, !dbg !17
  %18 = getelementptr float, ptr addrspace(1) %16, i64 %17, !dbg !17
  %19 = and i1 %12, %13, !dbg !14
  %20 = getelementptr float, ptr addrspace(3) @global_smem, i64 %17, !dbg !18
  %21 = select i1 %19, i32 8, i32 0, !dbg !18
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x8, $2;", "r,l,r,b"(ptr addrspace(3) %20, ptr addrspace(1) %18, i32 %21, i1 true) #3, !dbg !18
  tail call void asm sideeffect "cp.async.commit_group ;", ""() #3, !dbg !18
  %invariant.gep2 = getelementptr float, ptr addrspace(1) %1, i64 %17, !dbg !14
  %invariant.gep4 = getelementptr float, ptr addrspace(1) %0, i64 %17, !dbg !14
  br i1 %13, label %.lr.ph, label %._crit_edge, !dbg !14

.lr.ph:                                           ; preds = %6
  %22 = lshr i32 %9, 5, !dbg !12
  %23 = and i32 %9, 31, !dbg !12
  %24 = sub i32 %4, %8
  %25 = icmp eq i32 %23, 0
  %26 = and i32 %22, 7
  %27 = zext nneg i32 %26 to i64
  %28 = getelementptr float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 2048), i64 %27
  %29 = icmp slt i32 %9, 8
  %30 = sext i32 %9 to i64
  %31 = getelementptr float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 2048), i64 %30
  %32 = and i32 %9, 7
  %33 = icmp eq i32 %32, 0
  %34 = and i1 %29, %33
  br label %35, !dbg !14

35:                                               ; preds = %.lr.ph, %35
  %36 = phi i32 [ -1, %.lr.ph ], [ %41, %35 ]
  %37 = phi i32 [ %7, %.lr.ph ], [ %135, %35 ]
  %38 = icmp slt i32 %37, %24, !dbg !14
  %39 = add i32 %36, 1, !dbg !14
  %40 = icmp ugt i32 %36, 2147483646, !dbg !14
  %41 = select i1 %40, i32 %39, i32 0, !dbg !14
  tail call void asm sideeffect "cp.async.wait_group 0x0;", ""() #3, !dbg !18
  tail call void @llvm.nvvm.barrier0(), !dbg !18
  %42 = shl i32 %41, 9, !dbg !18
  %43 = sext i32 %42 to i64, !dbg !18
  %gep = getelementptr float, ptr addrspace(3) %20, i64 %43, !dbg !18
  %44 = load float, ptr addrspace(3) %gep, align 8, !dbg !18
  %45 = getelementptr inbounds i8, ptr addrspace(3) %gep, i64 4, !dbg !18
  %46 = load float, ptr addrspace(3) %45, align 4, !dbg !18
  %47 = select i1 %12, float %44, float 0xFFF0000000000000, !dbg !18
  %48 = select i1 %12, float %46, float 0xFFF0000000000000, !dbg !18
  %49 = tail call float @llvm.maxnum.f32(float %47, float %48), !dbg !19
  %50 = bitcast float %49 to i32, !dbg !24
  %51 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %50, i32 16, i32 31), !dbg !24
  %52 = bitcast i32 %51 to float, !dbg !24
  %53 = tail call float @llvm.maxnum.f32(float %49, float %52), !dbg !19
  %54 = bitcast float %53 to i32, !dbg !24
  %55 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %54, i32 8, i32 31), !dbg !24
  %56 = bitcast i32 %55 to float, !dbg !24
  %57 = tail call float @llvm.maxnum.f32(float %53, float %56), !dbg !19
  %58 = bitcast float %57 to i32, !dbg !24
  %59 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %58, i32 4, i32 31), !dbg !24
  %60 = bitcast i32 %59 to float, !dbg !24
  %61 = tail call float @llvm.maxnum.f32(float %57, float %60), !dbg !19
  %62 = bitcast float %61 to i32, !dbg !24
  %63 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %62, i32 2, i32 31), !dbg !24
  %64 = bitcast i32 %63 to float, !dbg !24
  %65 = tail call float @llvm.maxnum.f32(float %61, float %64), !dbg !19
  %66 = bitcast float %65 to i32, !dbg !24
  %67 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %66, i32 1, i32 31), !dbg !24
  %68 = bitcast i32 %67 to float, !dbg !24
  %69 = tail call float @llvm.maxnum.f32(float %65, float %68), !dbg !19
  %70 = bitcast float %69 to <1 x i32>, !dbg !24
  tail call void asm sideeffect "@$2 st.shared.b32 [ $0 + 0 ], $1;", "r,r,b"(ptr addrspace(3) %28, <1 x i32> %70, i1 %25) #3, !dbg !24
  tail call void @llvm.nvvm.barrier0(), !dbg !24
  %71 = tail call i32 asm sideeffect "@$2 ld.shared.b32 $0, [ $1 + 0 ];", "=r,r,b"(ptr addrspace(3) %31, i1 %29) #3, !dbg !24
  %72 = bitcast i32 %71 to float, !dbg !24
  %73 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %71, i32 4, i32 31), !dbg !24
  %74 = bitcast i32 %73 to float, !dbg !24
  %75 = tail call float @llvm.maxnum.f32(float %72, float %74), !dbg !19
  %76 = bitcast float %75 to i32, !dbg !24
  %77 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %76, i32 2, i32 31), !dbg !24
  %78 = bitcast i32 %77 to float, !dbg !24
  %79 = tail call float @llvm.maxnum.f32(float %75, float %78), !dbg !19
  %80 = bitcast float %79 to i32, !dbg !24
  %81 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %80, i32 1, i32 31), !dbg !24
  %82 = bitcast i32 %81 to float, !dbg !24
  %83 = tail call float @llvm.maxnum.f32(float %79, float %82), !dbg !19
  %84 = bitcast float %83 to <1 x i32>, !dbg !24
  tail call void asm sideeffect "@$2 st.shared.b32 [ $0 + 0 ], $1;", "r,r,b"(ptr addrspace(3) %31, <1 x i32> %84, i1 %34) #3, !dbg !24
  tail call void @llvm.nvvm.barrier0(), !dbg !24
  %85 = load float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 2048), align 16, !dbg !24
  %86 = fsub float %47, %85, !dbg !25
  %87 = fsub float %48, %85, !dbg !25
  %88 = fmul float %86, 0x3FF7154760000000, !dbg !26
  %89 = tail call float asm "ex2.approx.f32 $0, $1;", "=f,f"(float %88) #3, !dbg !26
  %90 = fmul float %87, 0x3FF7154760000000, !dbg !26
  %91 = tail call float asm "ex2.approx.f32 $0, $1;", "=f,f"(float %90) #3, !dbg !26
  tail call void @llvm.nvvm.barrier0(), !dbg !27
  %92 = fadd float %89, %91, !dbg !29
  %93 = bitcast float %92 to i32, !dbg !27
  %94 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %93, i32 16, i32 31), !dbg !27
  %95 = bitcast i32 %94 to float, !dbg !27
  %96 = fadd float %92, %95, !dbg !29
  %97 = bitcast float %96 to i32, !dbg !27
  %98 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %97, i32 8, i32 31), !dbg !27
  %99 = bitcast i32 %98 to float, !dbg !27
  %100 = fadd float %96, %99, !dbg !29
  %101 = bitcast float %100 to i32, !dbg !27
  %102 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %101, i32 4, i32 31), !dbg !27
  %103 = bitcast i32 %102 to float, !dbg !27
  %104 = fadd float %100, %103, !dbg !29
  %105 = bitcast float %104 to i32, !dbg !27
  %106 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %105, i32 2, i32 31), !dbg !27
  %107 = bitcast i32 %106 to float, !dbg !27
  %108 = fadd float %104, %107, !dbg !29
  %109 = bitcast float %108 to i32, !dbg !27
  %110 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %109, i32 1, i32 31), !dbg !27
  %111 = bitcast i32 %110 to float, !dbg !27
  %112 = fadd float %108, %111, !dbg !29
  %113 = bitcast float %112 to <1 x i32>, !dbg !27
  tail call void asm sideeffect "@$2 st.shared.b32 [ $0 + 0 ], $1;", "r,r,b"(ptr addrspace(3) %28, <1 x i32> %113, i1 %25) #3, !dbg !27
  tail call void @llvm.nvvm.barrier0(), !dbg !27
  %114 = tail call i32 asm sideeffect "@$2 ld.shared.b32 $0, [ $1 + 0 ];", "=r,r,b"(ptr addrspace(3) %31, i1 %29) #3, !dbg !27
  %115 = bitcast i32 %114 to float, !dbg !27
  %116 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %114, i32 4, i32 31), !dbg !27
  %117 = bitcast i32 %116 to float, !dbg !27
  %118 = fadd float %115, %117, !dbg !29
  %119 = bitcast float %118 to i32, !dbg !27
  %120 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %119, i32 2, i32 31), !dbg !27
  %121 = bitcast i32 %120 to float, !dbg !27
  %122 = fadd float %118, %121, !dbg !29
  %123 = bitcast float %122 to i32, !dbg !27
  %124 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %123, i32 1, i32 31), !dbg !27
  %125 = bitcast i32 %124 to float, !dbg !27
  %126 = fadd float %122, %125, !dbg !29
  %127 = bitcast float %126 to <1 x i32>, !dbg !27
  tail call void asm sideeffect "@$2 st.shared.b32 [ $0 + 0 ], $1;", "r,r,b"(ptr addrspace(3) %31, <1 x i32> %127, i1 %34) #3, !dbg !27
  tail call void @llvm.nvvm.barrier0(), !dbg !27
  %128 = load float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 2048), align 16, !dbg !27
  %129 = tail call float asm "div.full.f32 $0, $1, $2;", "=r,r,r"(float %89, float %128) #3, !dbg !30
  %130 = tail call float asm "div.full.f32 $0, $1, $2;", "=r,r,r"(float %91, float %128) #3, !dbg !30
  %131 = mul i32 %37, %3, !dbg !31
  %132 = sext i32 %131 to i64, !dbg !32
  %gep3 = getelementptr float, ptr addrspace(1) %invariant.gep2, i64 %132, !dbg !33
  %133 = bitcast float %129 to i32, !dbg !34
  %134 = bitcast float %130 to i32, !dbg !34
  tail call void asm sideeffect "@$3 st.global.v2.b32 [ $2 + 0 ], { $0, $1 };", "r,r,l,b"(i32 %133, i32 %134, ptr addrspace(1) %gep3, i1 %12) #3, !dbg !34
  %135 = add i32 %37, %8, !dbg !14
  %136 = mul i32 %135, %2, !dbg !15
  %137 = sext i32 %136 to i64, !dbg !16
  %gep5 = getelementptr float, ptr addrspace(1) %invariant.gep4, i64 %137, !dbg !17
  %138 = and i1 %12, %38, !dbg !14
  %139 = select i1 %138, i32 8, i32 0, !dbg !18
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x8, $2;", "r,l,r,b"(ptr addrspace(3) %20, ptr addrspace(1) %gep5, i32 %139, i1 true) #3, !dbg !18
  tail call void asm sideeffect "cp.async.commit_group ;", ""() #3, !dbg !18
  %140 = icmp slt i32 %135, %4, !dbg !14
  br i1 %140, label %35, label %._crit_edge, !dbg !14

._crit_edge:                                      ; preds = %35, %6
  tail call void asm sideeffect "cp.async.wait_group 0x0;", ""() #3, !dbg !14
  tail call void @llvm.nvvm.barrier0(), !dbg !14
  ret void, !dbg !35
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier0() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #0

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.bfly.i32(i32, i32, i32, i32) #2

attributes #0 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { convergent nocallback nounwind }
attributes #2 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}
!nvvm.annotations = !{!4, !5}
!llvm.ident = !{!6}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 1}
!2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "triton", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!3 = !DIFile(filename: "inspect_softmax_ir.py", directory: "/home/ubuntu/Learn-Triton/05_fused_softmax/practice")
!4 = !{ptr @_softmax_kernel, !"kernel", i32 1}
!5 = !{ptr @_softmax_kernel, !"reqntidx", i32 256}
!6 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!7 = distinct !DISubprogram(name: "_softmax_kernel", linkageName: "_softmax_kernel", scope: !3, file: !3, line: 45, type: !8, scopeLine: 45, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!8 = !DISubroutineType(cc: DW_CC_normal, types: !9)
!9 = !{}
!10 = !DILocation(line: 52, column: 30, scope: !7)
!11 = !DILocation(line: 53, column: 32, scope: !7)
!12 = !DILocation(line: 56, column: 37, scope: !7)
!13 = !DILocation(line: 58, column: 38, scope: !7)
!14 = !DILocation(line: 54, column: 57, scope: !7)
!15 = !DILocation(line: 55, column: 46, scope: !7)
!16 = !DILocation(line: 55, column: 36, scope: !7)
!17 = !DILocation(line: 57, column: 40, scope: !7)
!18 = !DILocation(line: 59, column: 32, scope: !7)
!19 = !DILocation(line: 163, column: 27, scope: !20, inlinedAt: !23)
!20 = distinct !DILexicalBlockFile(scope: !22, file: !21, discriminator: 0)
!21 = !DIFile(filename: "standard.py", directory: "/home/ubuntu/.local/lib/python3.10/site-packages/triton/language")
!22 = distinct !DILexicalBlockFile(scope: !7, file: !21, discriminator: 0)
!23 = !DILocation(line: 60, column: 37, scope: !7)
!24 = !DILocation(line: 184, column: 40, scope: !22, inlinedAt: !23)
!25 = !DILocation(line: 60, column: 30, scope: !7)
!26 = !DILocation(line: 61, column: 31, scope: !7)
!27 = !DILocation(line: 267, column: 36, scope: !22, inlinedAt: !28)
!28 = !DILocation(line: 62, column: 31, scope: !7)
!29 = !DILocation(line: 256, column: 15, scope: !20, inlinedAt: !28)
!30 = !DILocation(line: 63, column: 37, scope: !7)
!31 = !DILocation(line: 64, column: 54, scope: !7)
!32 = !DILocation(line: 64, column: 44, scope: !7)
!33 = !DILocation(line: 65, column: 40, scope: !7)
!34 = !DILocation(line: 65, column: 53, scope: !7)
!35 = !DILocation(line: 54, column: 4, scope: !7)
